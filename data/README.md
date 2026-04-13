# Data Notes

This directory stores training data used by `MicroLM`.

Current contents:

- `smoke/`: deterministic tiny token arrays used to validate the training loop and config wiring
- `pretrain_hq.jsonl`: downloaded MiniMind pretraining corpus (`{"text": ...}` JSONL)
- `pretrain/`: generated plain-text train/valid splits plus tokenizer corpus metadata

Planned later:

- pretraining corpus metadata
- tokenizer outputs
- SFT datasets and prompt templates
