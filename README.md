# Hystar

Compact reference code for **Hystar: Hypernetwork-Driven Style-Adaptive Retrieval via Dynamic SVD Modulation** (ICLR 2026).

Hystar adapts a frozen vision-language encoder to heterogeneous query styles by combining:

- **Dynamic SVD modulation** on selected attention layers, where a DINOv2 style embedding conditions a lightweight hypernetwork that predicts singular-value updates.
  For CLIP self-attention, the packed projection `W_qkv = [W_q; W_k; W_v]` is decomposed jointly, so one predicted `delta_s` reconstructs one full `[3d, d]` QKV update.
- **Static SVD modulation** on MLP layers for style-independent cross-domain calibration.
- **StyleNCE**, an OT-weighted contrastive objective for cross-style retrieval.

This repository is intentionally compact: it keeps the CLIP reference path and the code needed for the paper's core method, training, DSR evaluation, DomainNet retrieval, and DomainNet zero-shot classification. Experimental/obsolete scripts, generated figures, local caches, vendored TorchMeta, a full DINOv2 source snapshot, and incomplete BLIP/ALBEF branches have been removed.

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

If GPU memory is limited, reduce `--batch-size` and increase `--grad-accum` accordingly.

Before a full run, you can optionally verify the corrected joint-QKV attention path with:

```bash
pip install pytest
pytest -q tests/test_joint_qkv.py
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

## What was intentionally removed

The original research directory contained many files that are not needed for a clean public release: `train_1.py`, `train_2.py`, IA3/baseline scripts, gamma/OT plotting studies, attention-map and t-SNE utilities, generated sample images, local path lists, TorchMeta, a full DINOv2 source copy, and BLIP/ALBEF code paths whose external dependencies were not included in the archive. They are not required to understand or run the core Hystar-CLIP implementation.

## Reproducibility notes

- StyleNCE is written in paper-aligned OT notation: `c_ij = -(1 - sim_ij)` and `K_ij = exp(-c_ij / lambda)`, which is algebraically identical to `exp((1 - sim_ij) / lambda)` used in the original research implementation. This rewrite does not change checkpoint compatibility or numerical behavior.
- The CLIP attention path uses **joint packed-QKV SVD**: SVD is applied directly to `in_proj_weight` (`[3d, d]`), and the hypernetwork predicts one singular-value update vector whose reconstruction updates Q/K/V jointly. This corrects an older source variant that decomposed only the Q block and repeated its reconstructed update three times. Because the spectral basis changes, checkpoints from that older variant should not be loaded into this corrected implementation; retrain from the base CLIP weights.
- The original source contained machine-specific absolute paths; all such paths were removed.
- DINOv2 is frozen and used only to obtain the style embedding.
- No pretrained weights or datasets are bundled.

## Citation

```bibtex
@inproceedings{cai2026hystar,
  title     = {Hystar: Hypernetwork-Driven Style-Adaptive Retrieval via Dynamic SVD Modulation},
  author    = {Yujia Cai and Boxuan Li and Chenghao Xu and Jiexi Yan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## License

Please add the license you want to use for the Hystar project before publishing the repository. Third-party license information for retained code is included in `THIRD_PARTY_NOTICES.md` and `third_party/`.
