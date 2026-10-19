# MoSID

Source code for the two-stage MoSID recommendation pipeline:

1. Train a unified, block-structured residual quantizer on item embeddings.
2. Export semantic item IDs (SIDs) and codebook weights.
3. Train the MoSID sequence-to-SID recommendation model.
4. Evaluate recommendations with trie-constrained beam search and rank fusion.

The supplied configuration uses the Movies dataset in `nm` mode. Preprocessed data are distributed separately; see [the data specification](docs/DATA.md).

## Environment

Target platform: Linux, Python 3.11, and an NVIDIA GPU. The default Stage 2 configuration enables BF16 autocast and therefore requires BF16-capable CUDA hardware. Training uses one selected GPU. The default data loader settings assume a machine with sufficient CPU and RAM resources.

The dependency versions in `requirements.txt` are a reference environment. This source release has been checked statically; dependency installation, Linux/CUDA execution, training, and numerical reproduction have not been validated for this release.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Use a CUDA wheel compatible with the machine's driver. TensorBoard is disabled in both supplied configurations. Install `tensorboard` separately if enabling `train.tensorboard`. The shared RQ-VAE module also contains an optional constrained-clustering helper using `k-means-constrained`; the supplied unified configuration uses the built-in masked Sinkhorn initialization instead.

## Data

The preprocessed Movies data are available on [Hugging Face Datasets](https://huggingface.co/datasets/AnonymousButt/MoSID-data).

Use the [fixed data version](https://huggingface.co/datasets/AnonymousButt/MoSID-data/tree/766b67a9640e29d7acfd37fddffbbc9036a4ad76) and follow the download and SHA-256 verification commands in [docs/DATA.md](docs/DATA.md).

The commands place the files under `datasets/amazon/Movies/nm/`, matching the supplied configurations.

Both YAML files use `data.base_path: "./datasets/amazon"`. The resolved data directory is `<base_path>/<dataset>/<mode>`. Run all commands below from the repository root. To use another location, pass the same `--data.base_path` override at every stage.

## Training and evaluation

### 1. Train the unified RQ-VAE

```bash
python train_rqvae.py --config configs/rqvae_unified.yaml
```

With the supplied seed and output directory, the selected checkpoint is:

```text
output/Movies_nm_rqvae_unified_seed_42/best.pt
```

The model uses three quantization layers, 256 codes per layer, and three 64-dimensional latent blocks in `collab`, `text`, `vision` order. Text and visual embeddings have dimension 768; collaborative embeddings have dimension 256. Initialization processes the full item catalog and can require substantial GPU memory.
### 2. Export SIDs and codebooks

```bash
python scripts/export_sid.py \
  --config configs/rqvae_unified.yaml \
  --checkpoint output/Movies_nm_rqvae_unified_seed_0/best.pt
```

This writes the following files into `datasets/amazon/Movies/nm/`:

- `sid_codes_unified.pt`: `[N+1, 3]` integer tensor; item codes are in `1..256`, while padding row zero contains zeros.
- `codebook_weights_unified.pt`: a list of three `[256, 192]` tensors, with contiguous collaborative, text, and visual blocks.
- `sid_unified_source.json`: the Stage 1 seed and checkpoint filename.

Stage 2 reads these artifacts. Keep the SID table, codebooks, and metadata together and regenerate them together when changing Stage 1.

### 3. Train MoSID

```bash
python train_mosid.py --config configs/mosid.yaml
```

The default micro-batch size is 512 with four gradient-accumulation steps. Stage 2 uses three branches initialized from the unified codebook blocks, sequence generation losses, memory orthogonalization, and raw compatibility supervision. The selected checkpoint is:

```text
output/Movies_nm_mosid_seed_42-42/best.pt
```

The first seed in this directory name comes from the exported Stage 1 metadata; the second is the Stage 2 training seed. Each run saves a resolved `config.yaml` and training logs. Existing complete seeded runs are reused.
To run a fresh experiment while preserving previous outputs, use a different `--train.output_dir` for training and update checkpoint paths accordingly.

### 4. Evaluate the selected checkpoint

```bash
python test_mosid.py \
  --checkpoint output/Movies_nm_mosid_seed_0-0/best.pt \
  --split test
```

Evaluation loads the configuration saved beside the checkpoint. The supplied configuration selects rank-fusion parameters on validation data before test evaluation (`rrf.gs_on_valid: true`). Test labels are used only for reporting test metrics. Training validation uses beam size 10; standalone evaluation uses beam size 50. The configured ranking cutoffs are 10 and 20. Evaluation logs are
written under the checkpoint directory in `test_beam_eval/`.
