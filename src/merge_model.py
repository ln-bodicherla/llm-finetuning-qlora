"""
Script to merge LoRA adapter weights into the base model.

Loads a base model in full precision, applies the trained LoRA adapter,
merges the weights, and saves the resulting model. Optionally pushes
the merged model to the Hugging Face Hub.

Usage:
    python -m src.merge_model \
        --adapter-path ./outputs/final \
        --output-path ./outputs/merged \
        --base-model meta-llama/Llama-2-7b-hf

    # With Hub upload
    python -m src.merge_model \
        --adapter-path ./outputs/final \
        --output-path ./outputs/merged \
        --push-to-hub \
        --hub-repo-id your-username/your-model-name
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def detect_base_model(adapter_path: str) -> Optional[str]:
    """Attempt to read the base model name from the adapter config.

    Args:
        adapter_path: Path to the adapter directory.

    Returns:
        Base model name if found, None otherwise.
    """
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    return config.get("base_model_name_or_path")


def merge_lora_weights(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    safe_serialization: bool = True,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_private: bool = False,
) -> str:
    """Merge LoRA adapter weights into the base model and save.

    This function performs the following steps:
    1. Load the base model in the specified precision
    2. Load and apply the LoRA adapter
    3. Merge adapter weights into the base model
    4. Unload the adapter (convert back to standard model)
    5. Save the merged model and tokenizer
    6. Optionally push to Hugging Face Hub

    Args:
        base_model_name: HuggingFace model ID or path for the base model.
        adapter_path: Path to the trained LoRA adapter directory.
        output_path: Directory to save the merged model.
        torch_dtype: Precision for the merged model.
        device_map: Device placement strategy.
        trust_remote_code: Whether to trust remote code.
        safe_serialization: Use safetensors format.
        push_to_hub: Whether to push to Hugging Face Hub.
        hub_repo_id: Hub repository ID (required if push_to_hub=True).
        hub_private: Whether the Hub repo should be private.

    Returns:
        Path to the saved merged model directory.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. "
            "Make sure this is a valid PEFT adapter directory."
        )

    logger.info("=" * 60)
    logger.info("  LoRA Weight Merging")
    logger.info("=" * 60)
    logger.info("Base model:  %s", base_model_name)
    logger.info("Adapter:     %s", adapter_path)
    logger.info("Output:      %s", output_path)
    logger.info("Dtype:       %s", torch_dtype)
    logger.info("Push to Hub: %s", push_to_hub)
    logger.info("-" * 60)

    logger.info("Step 1/5: Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    total_params = sum(p.numel() for p in base_model.parameters())
    logger.info("Base model loaded: %s parameters", f"{total_params:,}")

    logger.info("Step 2/5: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Step 3/5: Loading and applying LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
    )

    with open(adapter_config) as f:
        lora_config = json.load(f)
    logger.info(
        "LoRA config: r=%s, alpha=%s, modules=%s",
        lora_config.get("r"),
        lora_config.get("lora_alpha"),
        lora_config.get("target_modules"),
    )

    logger.info("Step 4/5: Merging weights...")
    merged_model = model.merge_and_unload(progressbar=True)

    merged_params = sum(p.numel() for p in merged_model.parameters())
    logger.info("Merged model: %s parameters", f"{merged_params:,}")

    logger.info("Step 5/5: Saving merged model...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=safe_serialization,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(output_dir))

    config_info = {
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "lora_r": lora_config.get("r"),
        "lora_alpha": lora_config.get("lora_alpha"),
        "target_modules": lora_config.get("target_modules"),
        "merge_dtype": str(torch_dtype),
    }
    merge_info_path = output_dir / "merge_info.json"
    with open(merge_info_path, "w") as f:
        json.dump(config_info, f, indent=2)

    saved_files = list(output_dir.glob("*"))
    total_size = sum(f.stat().st_size for f in saved_files if f.is_file())
    logger.info(
        "Saved %d files (%.2f GB) to %s",
        len(saved_files), total_size / (1024**3), output_dir,
    )

    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("--hub-repo-id is required when --push-to-hub is set")

        logger.info("Pushing to Hugging Face Hub: %s", hub_repo_id)
        merged_model.push_to_hub(
            hub_repo_id,
            safe_serialization=safe_serialization,
            private=hub_private,
        )
        tokenizer.push_to_hub(hub_repo_id, private=hub_private)
        logger.info("Successfully pushed to Hub: https://huggingface.co/%s", hub_repo_id)

    logger.info("-" * 60)
    logger.info("Merge complete!")
    logger.info("Merged model saved to: %s", output_dir)

    return str(output_dir)


def main():
    """CLI entry point for model merging."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights into base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapter-path", type=str, required=True,
        help="Path to the trained LoRA adapter directory",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model name (auto-detected from adapter config if not provided)",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision for merged weights (default: float16)",
    )
    parser.add_argument(
        "--device-map", type=str, default="auto",
        help="Device placement strategy (default: auto)",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code in model definition",
    )
    parser.add_argument(
        "--no-safe-serialization", action="store_true",
        help="Save in PyTorch .bin format instead of safetensors",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push merged model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub-repo-id", type=str, default=None,
        help="Hugging Face Hub repository ID (e.g., user/model-name)",
    )
    parser.add_argument(
        "--hub-private", action="store_true",
        help="Make the Hub repository private",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_model = args.base_model
    if base_model is None:
        base_model = detect_base_model(args.adapter_path)
        if base_model is None:
            logger.error(
                "Could not detect base model. Provide --base-model explicitly."
            )
            sys.exit(1)
        logger.info("Auto-detected base model: %s", base_model)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    merge_lora_weights(
        base_model_name=base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        safe_serialization=not args.no_safe_serialization,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_private=args.hub_private,
    )


if __name__ == "__main__":
    main()
