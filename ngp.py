import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import utils
from matplotlib import pyplot as plt
import tqdm
import numpy as np
import einops


class NeuralGraphicsPrimitiveModel(pl.LightningModule):
    def __init__(self, mlp, dimension=2, levels=16, max_entries_per_level=2 ** 16, feature_dim=2, min_resolution=16,
                 max_resolution=16384, **kwargs):
        super().__init__(**kwargs)
        self.mlp = mlp
        self.d = dimension
        self.L = levels
        self.T = max_entries_per_level
        self.F = feature_dim
        self.N_min = min_resolution
        self.N_max = max_resolution

        b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.L - 1))
        N_l = torch.Tensor([np.floor(self.N_min * (b ** l)) for l in range(self.L)]).detach()
        self.register_buffer("N_l", N_l)
        self.create_hashmap_parameters()

        self.ema = utils.ExponentialMovingAverage(self.parameters(), decay=0.95)

    def create_hashmap_parameters(self) -> None:
        feat_init_min = -10 ** -4
        feat_init_max = 10 ** -4
        shape = (self.L, self.T, self.F)
        feats = ((feat_init_max - feat_init_min) * torch.rand(shape, device=self.device)) + feat_init_min
        self.register_parameter("features", torch.nn.Parameter(feats))

    def get_hypercube_vertices(self, low_coords_bld, high_coords_bld):
        b, l, d = low_coords_bld.shape
        coords_bl2d = torch.stack((low_coords_bld, high_coords_bld), dim=2)
        indices_blDd = einops.repeat(torch.cartesian_prod(*([torch.arange(2, device=self.device)] * d)),
                                     "v d -> b l v d",
                                     b=b, l=l)
        vertices_blDd = torch.gather(coords_bl2d, 2, indices_blDd)
        return vertices_blDd

    def interpolate(self, scaled_coordinates_bld, feats_blDf, vertices_blDd, smooth=False):
        """
        :param scaled_coordinates_bld:
        :param feats_blDf:
        :param vertices_blDd:
        :return:
        """
        b, l, d = scaled_coordinates_bld.shape
        # get cube side length, when the coordinates are scaled all voxels
        # have side length one
        side_lengths_bl1d = torch.ones((b, l, 1, d), device=self.device)
        # n-linear interpolation can be taken as the vertex's value times the volume of the
        # n-dimensional volume with corners defined by the *opposite* vertex and the point in the interior
        residuals_blDd = torch.clamp(
            side_lengths_bl1d - torch.abs(vertices_blDd - scaled_coordinates_bld.view(b, l, 1, d)),
            min=0.0001,
            max=0.9999
        )
        # the volume is obviously the reduction along that dimension via multiplication
        weights_blD1 = einops.reduce(residuals_blDd,
                                     "b l D d -> b l D 1",
                                     "prod")
        if smooth:
            weights_blD1 = (weights_blD1 ** 2) * (3.0 - (2.0 * weights_blD1))
        # multiply each vertex value by the weights and sum along the vertices
        interpolated_feats_blf = einops.reduce(feats_blDf * weights_blD1,
                                               "b l D f -> b l f",
                                               "sum")
        return interpolated_feats_blf

    def get_interpolated_features(self, coords, smooth=False):
        """
        :param x: b x d position vector in [0,1] for each dimension
        :return: b x (L F) features to use as input to the network
        """
        scaled_coords_bld = torch.einsum("bd,l->bld", coords, self.N_l)
        if smooth:
            # add half voxel size
            scaled_coords_bld += 1.0 / (2.0 * einops.rearrange(self.N_l, "l -> 1 l 1"))
        low_coords_bld = torch.floor(scaled_coords_bld).long()
        # add a bit to make sure we round up
        high_coords_bld = torch.ceil(scaled_coords_bld + (1.0 / (self.N_max + 1))).long()
        vertices_blDd = self.get_hypercube_vertices(low_coords_bld, high_coords_bld)
        b, l, D, d = vertices_blDd.shape

        feat_indices_lN = einops.rearrange(utils.spatial_hash(vertices_blDd.view(b * l * D, d), self.T),
                                           "(b l D) -> l (b D)", b=b, l=l, D=D)
        l_indices = torch.arange(l, dtype=torch.long, device=self.device)
        feats_blDf = einops.rearrange(self.features[l_indices[:, None], feat_indices_lN, :],
                                      "l (b D) f -> b l D f", b=b, l=l, D=D)
        interpolated_feats_blf = self.interpolate(scaled_coords_bld, feats_blDf, vertices_blDd, smooth=smooth)

        return interpolated_feats_blf.flatten(start_dim=1)

    def forward(self, x):
        coords, eta = x[:, :self.d], x[:, self.d:]
        interpolated_feats_bF = self.get_interpolated_features(coords, smooth=True)
        final_feats_bF = torch.cat((interpolated_feats_bF, eta), dim=1)
        return self.mlp(final_feats_bF)

    def step(self, batch, batch_idx, phase):
        raise NotImplementedError("Implement this for various methods!")

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx, "training")

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx, "validation")

    def configure_optimizers(self):
        self.ema.to(self.device)
        return torch.optim.AdamW(self.parameters(), betas=(0.9, 0.99), eps=10e-15, weight_decay=10e-6)

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.parameters())


class SDFNGPModel(NeuralGraphicsPrimitiveModel):
    def __init__(self, pos_enc_freqs=6, coords_min=-1.0, coords_max=1.0):
        mlp = utils.make_mlp(3 * 2 * pos_enc_freqs, 1, hidden_dim=128, hidden_layers=4)
        super().__init__(mlp, dimension=3, feature_dim=2)
        self.coords_min = coords_min
        self.coords_max = coords_max
        self.pos_enc_freqs = pos_enc_freqs

    def forward(self, x):
        # coords are between [coords_min, coords_max], scale to be between [0, 1]
        x -= self.coords_min
        x /= (self.coords_max - self.coords_min)
        pos_enc_x = utils.pos_encoding(x, self.pos_enc_freqs, dim=1)
        return super().forward(pos_enc_x)

    def step(self, batch, batch_idx, phase):
        x, d = batch
        outputs = self(x)
        loss = F.l1_loss(outputs.squeeze(), d.squeeze())
        self.log(f"{phase}/loss", loss)
        return loss


class GigapixelNGPModel(NeuralGraphicsPrimitiveModel):
    def __init__(self, pos_enc_freqs=6, coords_min=0.0, coords_max=1.0):
        mlp = utils.make_mlp(2 * 2 * pos_enc_freqs, 3, hidden_dim=64, hidden_layers=2,
                             output_nonlinearity=torch.nn.Sigmoid())
        super().__init__(mlp, dimension=2, max_resolution=4096, max_entries_per_level=2**18, feature_dim=2)
        self.coords_min = coords_min
        self.coords_max = coords_max
        self.pos_enc_freqs = pos_enc_freqs

    def forward(self, x):
        # coords are between [coords_min, coords_max], scale to be between [0, 1]
        x -= self.coords_min
        x /= (self.coords_max - self.coords_min)
        pos_enc_x = utils.pos_encoding(x, self.pos_enc_freqs, dim=1)
        return super().forward(pos_enc_x)

    def step(self, batch, batch_idx, phase):
        x, y = batch
        outputs = self(x)
        loss = F.mse_loss(outputs.squeeze(), y.squeeze())
        self.log(f"{phase}/loss", loss)
        self.log(f"{phase}/psnr", 20 * torch.log10(torch.Tensor([1.0]).to(self.device)) - 10 * torch.log10(loss))
        return loss
