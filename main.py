import time

import click
import torch
import torch.utils.data
import pytorch_lightning as pl
from ngp import *
import utils
from datasets import *
import trimesh
import skimage
import PIL
import matplotlib.pyplot as plt


def train_sdf(input_path, batch_size, output_path=None, model_path=None):
    num_freqs = 6
    sdf_path = pathlib.Path(input_path)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if sdf_path.suffix == ".obj":
        if model_path is None:
            train_dataset = OBJDataset(str(sdf_path), num_samples=2 ** 22, tsdf_min=-0.25, tsdf_max=0.25,
                                       voxel_resolution=512)
        val_dataset = OBJDataset(str(sdf_path), num_samples=2 ** 16, tsdf_min=-0.25, tsdf_max=0.25,
                                 voxel_resolution=512)
    elif sdf_path.suffix == ".sdf":
        if model_path is None:
            train_dataset = SDFDataset(str(sdf_path), tsdf_min=-0.25, tsdf_max=0.25)
        val_dataset = SDFDataset(str(sdf_path), tsdf_min=-0.25, tsdf_max=0.25)
    else:
        raise ValueError(f"Unsupported dataset type {sdf_path.suffix}")

    if model_path is None:
        model = SDFNGPModel(pos_enc_freqs=num_freqs)
    else:
        model = SDFNGPModel.load_from_checkpoint(model_path)
    model.to(dev)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    if model_path is None:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        trainer = pl.Trainer(
            gpus=1,
            logger=True,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="validation/loss", mode="min", patience=5),
                pl.callbacks.ModelCheckpoint(monitor="validation/loss", mode="min"),
            ],
            enable_checkpointing=True,
            min_epochs=1,
            max_epochs=50,
            precision=32 if dev == torch.device("cpu") else 16
        )

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    out_path = output_path or sdf_path.parent / (sdf_path.stem + "_pred.ply")
    predict_mesh(val_dataset, model, str(out_path))


def train_gigapixel(input_path, batch_size, output_path=None, model_path=None):
    num_freqs = 6
    img_path = pathlib.Path(input_path)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if img_path.suffix == ".exr":
        train_dataset = EXRDataset(str(img_path))
    else:
        train_dataset = ImageDataset(str(img_path))

    if model_path is None:
        model = GigapixelNGPModel(pos_enc_freqs=num_freqs)
    else:
        model = GigapixelNGPModel.load_from_checkpoint(model_path)
    model.to(dev)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    if model_path is None:
        trainer = pl.Trainer(
            gpus=1,
            logger=True,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="training/loss", mode="min", patience=5),
                pl.callbacks.ModelCheckpoint(monitor="training/loss", mode="min"),
            ],
            enable_checkpointing=True,
            min_epochs=1,
            max_epochs=50,
            precision=32 if dev == torch.device("cpu") else 16
        )

        trainer.fit(model, train_dataloaders=train_dataloader)

    out_path = output_path or img_path.parent / (img_path.stem + "_pred.jpg")
    predict_image(train_dataset, model, str(out_path), batch_size=batch_size)


def predict_mesh(dataset, model, filename):
    preds = []

    with torch.inference_mode():
        model.eval()
        t0 = time.monotonic()

        for coord_slice_WD3 in dataset.voxel_coords_hwd3:
            coord_slice_N3 = coord_slice_WD3.reshape(-1, coord_slice_WD3.shape[2]).to(model.device)
            pred_slice_N1 = model(coord_slice_N3)
            preds.append(pred_slice_N1.reshape(coord_slice_WD3.shape[:-1]).cpu())

        sdf_pred_hwd = torch.stack(preds, dim=0).squeeze()
        t1 = time.monotonic()
        print(f"Predicted SDF in {t1 - t0} seconds")

        mask = F.interpolate(
            einops.rearrange(torch.from_numpy(
                skimage.morphology.ball(sdf_pred_hwd.shape[0] // 2)),
                             "h w d -> 1 1 h w d"),
            sdf_pred_hwd.shape,
            mode="nearest",
        ).squeeze().bool().cpu().numpy()

        verts, faces, norms, _ = skimage.measure.marching_cubes(
            sdf_pred_hwd.numpy(),
            allow_degenerate=False,
            mask=mask,
            level=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=norms)
        trimesh.exchange.export.export_mesh(mesh, filename, "ply")
        print(f"Exported mesh to {filename}")


def predict_image(dataset, model, filename, batch_size=4096):
    pred_img = torch.zeros(dataset.rgb_hw3.shape, dtype=torch.float)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    coords_multiplier = torch.Tensor([dataset.rgb_hw3.shape[0] - 1, dataset.rgb_hw3.shape[1] - 1]).unsqueeze(0)

    with torch.inference_mode():
        model.eval()

        for batch in dataloader:
            x, _ = batch
            c = model(x.to(model.device))
            i = torch.round(x * coords_multiplier).long().cpu()
            pred_img[i[:, 0], i[:, 1]] = c.cpu()

    pred_img = torch.clamp(pred_img, min=0.0, max=1.0)
    PIL.Image.fromarray((pred_img.numpy() * 255.0).astype(np.uint8)).save(filename, quality=100)
    print(f"PSNR: {(20 * np.log10(1.0)) - (10 * np.log10(F.mse_loss(dataset.rgb_hw3.cpu(), pred_img).item()))}")


@click.command()
@click.option("--input-data", type=click.Path(), required=True, help="Path to input data")
@click.option("--task", type=click.Choice(["sdf", "gigapixel"]), required=True, help="Task to perform")
@click.option("--batch-size", type=click.INT, default=4096, help="Batch size")
@click.option("--output-path", type=click.Path(), required=False, default=None,
              help="Output path for generated artifacts")
@click.option("--model-path", type=click.Path(), required=False, default=None,
              help="Path of pretrained model to run inference with")
def main(
        input_data,
        task,
        batch_size,
        output_path,
        model_path,
):
    if task == "gigapixel":
        train_gigapixel(input_data, batch_size, output_path, model_path)
    elif task == "sdf":
        train_sdf(input_data, batch_size, output_path, model_path)
    else:
        raise ValueError(f"Unsupported task {task}")


if __name__ == "__main__":
    main()

