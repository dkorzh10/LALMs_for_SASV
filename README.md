# LALMs for SASV

Code release for the paper
**[Large Audio Language Models for Spoofing-Aware Speaker Verification](https://arxiv.org/abs/2607.14753)**
(SLT 2026).

Training and evaluation code for LALM-based spoofing-aware speaker verification
(SASV), covering:

- hard-label LoRA supervised fine-tuning (SFT)
- composite objectives with speaker (AAM / ArcFace) and spoof (BCE) auxiliary heads
- hard-sample mining
- chain-of-thought (reasoning) SFT
- GRPO-based reinforcement learning
- conventional ECAPA2 + Wav2Vec2-AASIST / WavLM fusion baselines

Primary LALM backbone: **SALMONN-7B**. Secondary: **Qwen2-Audio-7B**.

## Setup

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

### Required assets

Download / place the following and set paths in a repo-root `.env` file
(see `.env.example`). `./scripts/train.sh` loads `.env` automatically; YAML configs
expand `${VAR}`:

| Variable                                                          | Purpose                                                                        |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `DATA_ROOT`                                                     | Directory with train/val/test pair JSONs (+ optional noise/RIR lists of paths) |
| `OUTPUT_DIR`                                                    | Experiment outputs (default:`./outputs`)                                     |
| `SALMONN_CKPT`                                                  | Pretrained SALMONN-7B checkpoint (`.pth`)                                    |
| `LLAMA_PATH`                                                    | Vicuna-7B / LLaMA tokenizer+weights directory                                  |
| `WHISPER_PATH`                                                  | Whisper-large-v2 directory                                                     |
| `BEATS_PATH`                                                    | BEATs checkpoint (`.pt`)                                                     |
| `QWEN_AUDIO_PATH`                                               | Qwen2-Audio-7B-Instruct (for`configs/qwen_sft.yaml`)                         |
| `SALMONN_SFT_CKPT`                                              | CoT-SFT checkpoint used to initialize GRPO                                     |
| `CM_CKPT` / `AASIST_REPO` / `XLSR_WEIGHTS` / `ECAPA_CKPT` | Fusion baselines                                                               |

Public starting points:

- SALMONN: [tsinghua-ee/SALMONN](https://github.com/bytedance/SALMONN) / associated checkpoints
- Vicuna-7B-v1.1, Whisper-large-v2, BEATs (AS2M)
- Qwen2-Audio-7B-Instruct (Hugging Face)
- ECAPA2: `Jenthe/ECAPA2`
- AASIST CM: external Wav2Vec2-AASIST training repository + XLSR weights

### Data

Experiments in the paper use **ASVspoof 5** (Track 2 / open condition) enrollment–trial
pairs, enriched with bona fide speech from **VoxCeleb** (no speaker overlap with
evaluation). Audio is resampled to 16 kHz; SALMONN concatenates enrollment and trial
with a 1 s silence gap.

Provide JSON/JSONL pair files under `DATA_ROOT` (see `src/dataloaders/dataset.py` for
the expected fields: enrollment/trial paths, `answer` / `gt` in `{yes,no,gen}`, and
optional `reasoning` / `Acoustic features` / `reasons` for CoT). A helper converter
lives at `src/utils/scripts/convert_asvspoof5_sasv.py`.

Noise / RIR path lists for augmentation (optional):

```text
${DATA_ROOT}/noise_list.txt
${DATA_ROOT}/rir_list.txt
```

## Running

`./scripts/train.sh <config.yaml> [NUM_GPUS]` loads repo-root `.env` automatically. Create it once:

```bash
cp .env.example .env
# edit .env with your DATA_ROOT / checkpoint paths
```

Then run:

```bash
# Hard-label LoRA SFT (L1)
./scripts/train.sh configs/salmon_hard_label.yaml 4

# L1 + AAM (L2) + spoof BCE (L3)
./scripts/train.sh configs/salmon_aux_heads.yaml 4

# + hard-sample mining
./scripts/train.sh configs/salmon_hard_mining.yaml 4

# Reasoning / CoT SFT
./scripts/train.sh configs/salmon_cot_sft.yaml 4

# GRPO (set SALMONN_SFT_CKPT in .env)
./scripts/train.sh configs/salmon_grpo.yaml 4

# Qwen2-Audio hard-label SFT (set QWEN_AUDIO_PATH in .env)
./scripts/train.sh configs/qwen_sft.yaml 4

# ECAPA2 + Wav2Vec2-AASIST fusion baseline (set CM_CKPT / AASIST_REPO / XLSR_WEIGHTS / ECAPA_CKPT in .env)
./scripts/train.sh configs/baseline_ecapa_aasist.yaml 1
```

ECAPA2 + WavLM **power fusion** baseline (paper Table II) is evaluated with
(paths from `.env` or the environment):

```bash
# optional: SASV_CASCADE_ROOT=... in .env
python src/utils/scripts/eval_sasv_cascade.py \
  --config configs/baseline_ecapa_wavlm_fusion.yaml \
  --asv_type ecapa2 --asv_model "${ECAPA_CKPT}" \
  --cm_model "${WAVLM_CM_CKPT}" \
  --dataset "${DATA_ROOT}/test.json" \
  --q 0.5 --outputs_root "${OUTPUT_DIR}"
```

Or call the runner directly (still needs the same variables in the environment;
`source .env` or use `train.sh`):

```bash
set -a && source .env && set +a
PYTHONPATH=. python -m src.runner --config configs/salmon_hard_label.yaml
```

## Configs

| Config                                       | Paper role                  |
| -------------------------------------------- | --------------------------- |
| `configs/salmon_hard_label.yaml`           | LoRA SFT, CE only (L1)      |
| `configs/salmon_aux_heads.yaml`            | L1 + AAM L2 + spoof L3      |
| `configs/salmon_hard_mining.yaml`          | + hard-sample mining        |
| `configs/salmon_cot_sft.yaml`              | Reasoning / CoT SFT         |
| `configs/salmon_grpo.yaml`                 | GRPO                        |
| `configs/qwen_sft.yaml`                    | Qwen2-Audio LoRA SFT        |
| `configs/baseline_ecapa_aasist.yaml`       | ECAPA2 + W2V-AASIST fusion  |
| `configs/baseline_ecapa_wavlm_fusion.yaml` | ECAPA2 + WavLM power fusion |

Hyperparameters in these YAMLs follow the paper where practical (LoRA rank 128,
α=256, dropout 0.05, λ₁=1, λ₂=0.5, λ₃=1, AdamW 1e-5 with 1000 warmup steps,
effective batch size ≈1024 via `batch_size × accum_grad_iters × num_gpus`).
Adjust GPU count / accumulation to match your hardware.

## LALM Prompts

Prompt templates are under `src/prompts/`:

- `sasv_prompt.json` — hard-label yes / no / gen
- `sasv_reasoning_prompts.json` — CoT / structured reasoning
- `judge_prompts.json` — optional LLM-as-judge for GRPO

## Plotting

After a training run, generate learning curves (`training_overview.png`),
validation metrics, and an HTML sample preview:

```bash
./scripts/plot_experiment.sh salmon_hard_label
# or a specific run:
./scripts/plot_experiment.sh salmon_hard_label run_2026_09_22_02_14
```

Plots are written to `outputs/<experiment>/<run>/plots/` (or `$OUTPUT_DIR/...`
if set in `.env`). The script also prints a short GT / prediction preview from
the latest `samples_validation_epoch_*.jsonl`.

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

Some optional tests require local checkpoints (`LLAMA_PATH`, `ENRICHED_TRACES_JSON`).

## Citation

```bibtex
@article{savelyeva2026lalms,
  title={Large Audio Language Models for Spoofing-Aware Speaker Verification},
  author={Savelyeva, Sofya and Perunova, Mariia and Kushnir, Evgeny and
          Dvirniak, Artem and Korzh, Dmitrii and Rogov, Oleg Y.},
  journal={arXiv preprint arXiv:2607.14753},
  year={2026}
}
```

## License

Released for research use with the accompanying paper.
