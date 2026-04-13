# Smoke Data

These files are intentionally tiny and deterministic.

Purpose:

- verify that `scripts/train_pretrain.py --config configs/pretrain_smoke.json` works end-to-end
- provide a stable baseline for quick regression checks after refactors

Files expected here:

- `train_ids.npy`
- `valid_ids.npy`
- `metadata.json`
- `tokenizer_corpus.txt`
- `train.txt`
- `valid.txt`
