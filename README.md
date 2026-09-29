## Overview

<p align="center">
  <img src="assets/pipeline.pdf" width="95%">
</p>

<p align="center">
  <em>Overview of Hystar. Hystar combines hypernetwork-driven dynamic SVD modulation on attention layers with static SVD modulation on MLP layers for style-adaptive retrieval.</em>
</p>


# Hystar

Code for **Hystar: Hypernetwork-Driven Style-Adaptive Retrieval via Dynamic SVD Modulation** (ICLR 2026).

Hystar adapts a frozen vision-language encoder to heterogeneous query styles by combining:

- **Dynamic SVD modulation** on selected attention layers, where a DINOv2 style embedding conditions a lightweight hypernetwork that predicts singular-value updates.
  For self-attention, the packed projection `W_qkv = [W_q; W_k; W_v]` is decomposed jointly, so one predicted `delta_s` reconstructs one full `[3d, d]` QKV update.
- **Static SVD modulation** on MLP layers for style-independent cross-domain calibration.
- **StyleNCE**, an OT-weighted contrastive objective for cross-style retrieval.



## Repository layout

```text
.
├── src/
│   ├── train.py                 # DSR training
│   ├── test.py                  # DSR paired retrieval evaluation
│   ├── test_domainnet.py        # DomainNet zero-shot category retrieval
│   ├── test_classification.py   # DomainNet zero-shot classification
│   ├── data/
│   │   ├── DSR.py
│   │   └── DomainNet.py
│   ├── model/
│   │   ├── hyper_net.py
│   │   ├── re_model.py
│   │   └── style_clip.py
│   └── utils/
│       ├── loss.py
│       └── utils.py
├── LoRACLIP/                    # minimal SVD attention/linear implementation
├── third_party/
├── requirements.txt
└── CITATION.bib
```

## Installation

```bash
conda create -n hystar python=3.10 -y
conda activate hystar
pip install -r requirements.txt
```

The first model construction loads DINOv2 through `torch.hub`. OpenCLIP model weights are loaded through the `--clip-pretrained` argument (default: `openai`). You may also pass a local checkpoint path if your OpenCLIP version supports it.

## DSR data layout

The loader expects the following structure:

```text
/path/to/DSR/
├── train.json
├── test.json
├── images/
├── sketch/
├── mosaic/      # low-resolution query branch in the provided research code
├── art/
└── text/
```

Each JSON record must contain an `image` field. Text-query evaluation also expects a `caption` field that points to a text file under `text/`.

## Training

The default command uses the hyperparameters reported for the CLIP backbone: batch size 48, 35 epochs, static branch learning rate `1e-3`, dynamic hypernetwork learning rate `1e-5`, dynamic injection at transformer blocks 4/7/10/13, `gamma=80`, `lambda=1`, and 50 Sinkhorn iterations.

```bash
python -m src.train \
  --dataset-root /path/to/DSR \
  --output-dir outputs/hystar_clip \
  --device cuda
```

## DSR evaluation

Style query:

```bash
python -m src.test \
  --dataset-root /path/to/DSR \
  --checkpoint outputs/hystar_clip/best.pt \
  --query-type style \
  --style sketch
```

For the low-resolution setting used by the original source tree, use `--style mosaic`. For text queries:

```bash
python -m src.test \
  --dataset-root /path/to/DSR \
  --checkpoint outputs/hystar_clip/best.pt \
  --query-type text
```

## DomainNet retrieval

The root should contain DomainNet style folders and split files such as `real_test.txt`, `sketch_test.txt`, etc.

```bash
python -m src.test_domainnet \
  --dataset-root /path/to/DomainNet \
  --checkpoint outputs/hystar_clip/best.pt \
  --style sketch
```

## DomainNet zero-shot classification

```bash
python -m src.test_classification \
  --dataset-root /path/to/DomainNet \
  --checkpoint outputs/hystar_clip/best.pt \
  --style sketch
```



## Citation

```bibtex
@inproceedings{ICLR2026_c4dbadd6,
 author = {Cai, Yujia and Li, Boxuan and Xu, Chenghao and Yan, Jiexi},
 booktitle = {International Conference on Learning Representations},
 editor = {C. Vondrick and B. Hariharan and C. Raffel and L. Pinto and D. Yang and A. Faust},
 pages = {121024--121052},
 title = {Hystar: Hypernetwork-driven Style-adaptive Retrieval via Dynamic SVD Modulation},
 url = {https://proceedings.iclr.cc/paper_files/paper/2026/file/c4dbadd63360f0b9a79b1c541995b6cd-Paper-Conference.pdf},
 volume = {2026},
 year = {2026}
}
```
