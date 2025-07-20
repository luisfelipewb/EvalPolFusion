## Evaluation of Polarimetric Fusion for Semantic Segmentation in Aquatic Environments

<div align="center"> 

<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" />
</a>
</div>

## Introduction
This repository accompanies our paper **“Evaluation of Polarimetric Fusion for Semantic Segmentation in Aquatic Environments.”**  It contains what is needed to reproduce the results and build upon them.

The following GIF previews four random PoTATO-Seg test frames. We show UNet outputs for RGB, DIF, and POL inputs. Faded collor marks the region of interest. In the bottom: blue = ground truth, red = prediction, magenta = overlap.


<p align="center">
    <img src="img/unet.gif" alt="UNet Architecture" width="600"/>
</p>


If you use this code, please cite the paper using the BibTeX entry below (see the **Citation** section).


## Updates
- 07/2025, Initial repository release (anonymous)


## Environment
```bash
# Clone this repo
git clone https://anonymous.4open.science/r/EvalPolFusion/
cd EvalPolFusion

# Create and activate venv
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install the baseline module
pip install -e baseline/src/bottleseg
# Export the path for the fusion models
export PYTHONPATH=$(pwd)/fusion:$PYTHONPATH
```

## Data preparation
This project relies on **PoTATO** (Polarimetric Traces of Afloat Trash Objects). The original PoTATO release targets object detection. We provide a semantic-segmentation remix used in this paper.

Download the dataset:
```bash
# replace the URL after the review process
wget -O potato_seg.tar.gz "<placeholder-link-to-PoTATO-Segmentation>"
mkdir -p data
tar -xzf potato_seg.tar.gz -C data
```


## Baseline

To train each modality individually
```bash
cd baseline
python train.py --device "cuda:0" --run_name "test" --model_type "unet" --modality "dif" --epochs 30 --bs 4
```

To run inference on each modality individually
```bash
cd baseline
python test.py --device "cuda:0" --run_name "test" --model_type "unet" --modality "dif"
```

Alternatively, use the bash wrapper script. `run_all.sh` iterates over every modality defined at the top of the script and executes the full training-validation-testing pipeline for each one. Open the script and edit the variables (e.g. model_types, modalities, etc) to suit your setup before running it.
```bash
cd baseline
./run_all.sh
```


## Fusion
Before training fusion models, download pretrained [SegFormer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5), such as fusion/checkpoints/pretrained/segformers/mit_b3.pth:
```text
fusion/checkpoints/pretrained/segformers
├── mit_b2.pth
├── mit_b3.pth
└── mit_b4.pth
```


Train example
```bash
cd fusion
torchrun --nproc_per_node=2 tools/inference.py ../runs/dummyrun/preds/RGBAD --cfg configs/potatoMultiModalityRGBAD.yaml
```

Test example
```bash
#Example to run inferences on a model using RGBAD modalities.
cd fusion
torchrun --nproc_per_node=2 tools/train_mm.py ../runs/dummyrun/preds/RGBAD --cfg configs/potatoMultiModalityRGBAD.yaml
```

(Optional) Download pretrained weights.

Use the following link to download all pretrained weights for all modalities. [Link](https://dummylink) 


## Citation
If you use our baselines and fusion benchmark, please cite the following:
```
TODO
```

## Acknowledgements
Our codebase is based on the following public Github repositories:
- [DELIVER](https://github.com/jamycheung/DELIVER)
- [MMSFormer](https://github.com/csiplab/MMSFormer)
- [StitchFusion](https://github.com/LiBingyu01/StitchFusion/)
- [PoTATO dataset](https://github.com/luisfelipewb/PoTATO/)