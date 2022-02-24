
## Overview
This is an implementation of [Instant NGP](https://nvlabs.github.io/instant-ngp/) using Pytorch Lightning.

Currently, the supported tasks are Neural gigapixel images and Neural SDF. NeRF is in progress.

## Setup

```shell
conda env update -f environment.yml
conda activate ngp
git lfs install && git lfs pull  # to get sample data
```

Additionally, you may want to build the [SDFGen tool](https://github.com/christopherbatty/SDFGen) if you wish to
train Neural SDFs. This is not required, but the resulting SDFs are of higher quality, with fewer artifacts.

## Usage

For gigapixel images, the script accepts either standard PNG/JPG images or uncompressed EXR images. For SDFs,
the script accepts either the output of the SDFGen tool (above) or an OBJ file. If an OBJ file is provided, the
SDF is directly sampled from the mesh.

```shell
Usage: main.py [OPTIONS]

Options:
  --input-data PATH       Path to input data  [required]
  --task [sdf|gigapixel]  Task to perform  [required]
  --batch-size INTEGER    Batch size
  --output-path PATH      Output path for generated artifacts
  --model-path PATH       Path of pretrained model to run inference with
  --help                  Show this message and exit.
```

### Quickstart

SDF:
```shell
python3 main.py --input-data data/sdf/dragon.obj --task sdf --batch-size 4096
```

Gigapixel:
```shell
python3 main.py --input-data data/image/yosemite.jpg --task gigapixel --batch-size 32768
```