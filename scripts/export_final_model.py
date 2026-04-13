#!/usr/bin/env python3
"""export_final_model.py — Merge LoRA adaptor into base Qwen model for deployment.

Merges the InstructIE LoRA adaptor into Qwen2.5-1.5B-Instruct base model
and exports a single, self-contained model directory ready for vLLM serving.

Usage:
    python scripts/export_final_model.py                          # defaults
    python scripts/export_final_model.py --adaptor best_adaptor   # use best val_loss
    python scripts/export_final_model.py --skip-if-exists         # skip if merged dir exists
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# ── Paths (relative to project root) ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = PROJECT_ROOT / "Qwen2.5-1.5B-Instruct"
QWEN_LORA_DIR = PROJECT_ROOT / "outputs" / "qwen_lora"
ADAPTOR_FINAL = QWEN_LORA_DIR / "adaptor_final"
ADAPTOR_BEST = QWEN_LORA_DIR / "best_adaptor"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen_lora_merged_final"


def check_dependencies():
    """Verify required packages are available."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from peft import PeftModel  # noqa: F401
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Please install: pip install transformers peft")
        sys.exit(1)


def export_merged_model(
    base_path: Path,
    adaptor_path: Path,
    output_dir: Path,
    skip_if_exists: bool = False,
) -> dict:
    """Load base + LoRA, merge, and save as single model.

    Returns metadata dict with export info.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # ── Skip if already exported ──
    if skip_if_exists and output_dir.exists():
        config_json = output_dir / "config.json"
        if config_json.exists():
            print(f"[SKIP] Merged model already exists at {output_dir}")
            # Return existing metadata
            meta_path = output_dir / "export_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {"status": "skipped_existing", "output_dir": str(output_dir)}

    # ── Validate inputs ──
    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")
    if not adaptor_path.exists():
        raise FileNotFoundError(f"Adaptor not found: {adaptor_path}")

    t0 = time.time()

    # ── Load tokenizer ──
    print(f"[1/4] Loading tokenizer from {base_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    # ── Load base model ──
    print(f"[2/4] Loading base model from {base_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )

    # ── Load & merge LoRA ──
    print(f"[3/4] Loading LoRA adaptor from {adaptor_path} and merging ...")
    model = PeftModel.from_pretrained(model, str(adaptor_path))
    model = model.merge_and_unload()

    # ── Save merged model ──
    print(f"[4/4] Saving merged model to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - t0

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Read adaptor config for provenance
    adaptor_config_path = adaptor_path / "adapter_config.json"
    adaptor_meta = {}
    if adaptor_config_path.exists():
        with open(adaptor_config_path, "r", encoding="utf-8") as f:
            ac = json.load(f)
            adaptor_meta = {
                "r": ac.get("r"),
                "lora_alpha": ac.get("lora_alpha"),
                "target_modules": ac.get("target_modules"),
                "peft_version": ac.get("peft_version"),
            }

    # Write export metadata
    metadata = {
        "status": "exported",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": str(base_path),
        "adaptor_path": str(adaptor_path),
        "output_dir": str(output_dir),
        "total_params": total_params,
        "elapsed_sec": round(elapsed, 1),
        "adaptor_config": adaptor_meta,
    }
    with open(output_dir / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*50}")
    print("EXPORT COMPLETE")
    print(f"{'='*50}")
    print(f"  Base model:     {base_path}")
    print(f"  Adaptor:        {adaptor_path}")
    print(f"  Output:         {output_dir}")
    print(f"  Total params:   {total_params:,}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"  LoRA config:    r={adaptor_meta.get('r')}, alpha={adaptor_meta.get('lora_alpha')}")
    print(f"\nOutput directory contents:")
    for item in sorted(output_dir.iterdir()):
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  {item.name:<30} {size_mb:>8.1f} MB")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export merged Qwen+LoRA model for deployment")
    parser.add_argument(
        "--base-model", type=Path, default=BASE_MODEL_PATH,
        help="Path to base Qwen model",
    )
    parser.add_argument(
        "--adaptor", type=str, default="adaptor_final",
        choices=["adaptor_final", "best_adaptor"],
        help="Which adaptor to use (default: adaptor_final)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip export if output directory already exists",
    )
    args = parser.parse_args()

    check_dependencies()

    adaptor_path = QWEN_LORA_DIR / args.adaptor
    metadata = export_merged_model(
        base_path=args.base_model,
        adaptor_path=adaptor_path,
        output_dir=args.output_dir,
        skip_if_exists=args.skip_if_exists,
    )

    print(f"\nMetadata saved to {args.output_dir / 'export_metadata.json'}")


if __name__ == "__main__":
    main()
