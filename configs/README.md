# Config Notes

These config files freeze the current baseline choices for `MicroLM`.

They are not fully wired into the training scripts yet. Right now they serve two purposes:

- make the intended baseline explicit
- prevent hyperparameters from drifting while the project is still stabilizing

The immediate next step is to teach `scripts/train_pretrain.py` to accept one of these config files directly.

