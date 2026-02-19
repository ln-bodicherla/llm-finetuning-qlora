"""
Model loading with quantization and LoRA adapter configuration.

Handles loading base models with 4-bit quantization via bitsandbytes,
preparing models for k-bit training, configuring LoRA adapters, and
merging adapter weights back into the base model.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

COMPUTE_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelLoader:
    """Loads and configures language models for QLoRA fine-tuning.

    Manages the full model setup pipeline: quantized base model loading,
    tokenizer configuration, LoRA adapter application, and post-training
    weight merging.

    Args:
        base_model: HuggingFace model identifier or local path.
        quantization_config: Dict with quantization parameters.
        lora_config: Dict with LoRA hyperparameters.
        device_map: Device placement strategy ('auto', specific device, or dict).
        trust_remote_code: Whether to trust remote code in model definitions.
        attn_implementation: Attention implementation ('eager', 'sdpa', 'flash_attention_2').
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        quantization_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        self.base_model_name = base_model
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation

        self.quant_config = quantization_config or {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

        self.lora_params = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": DEFAULT_TARGET_MODULES,
            "task_type": "CAUSAL_LM",
        }

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.peft_model: Optional[PeftModel] = None

    def _build_bnb_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig from quantization parameters."""
        compute_dtype_str = self.quant_config.get("bnb_4bit_compute_dtype", "float16")
        compute_dtype = COMPUTE_DTYPE_MAP.get(compute_dtype_str, torch.float16)

        config = BitsAndBytesConfig(
            load_in_4bit=self.quant_config.get("load_in_4bit", True),
            load_in_8bit=self.quant_config.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=self.quant_config.get("bnb_4bit_use_double_quant", True),
        )

        logger.info(
            "Quantization config: 4bit=%s, dtype=%s, quant_type=%s, double_quant=%s",
            config.load_in_4bit,
            compute_dtype_str,
            config.bnb_4bit_quant_type,
            config.bnb_4bit_use_double_quant,
        )
        return config

    def load_base_model(self) -> PreTrainedModel:
        """Load the base model with quantization.

        Returns:
            Quantized PreTrainedModel ready for LoRA adapter application.
        """
        logger.info("Loading base model: %s", self.base_model_name)

        bnb_config = self._build_bnb_config()

        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "use_cache": False,
        }

        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )

        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )

        trainable_params, total_params = self._count_parameters(self.model)
        logger.info(
            "Base model loaded: %s | Total params: %s | Trainable: %s (%.2f%%)",
            self.base_model_name,
            f"{total_params:,}",
            f"{trainable_params:,}",
            100 * trainable_params / total_params if total_params > 0 else 0,
        )

        return self.model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the tokenizer with proper padding setup.

        Returns:
            Configured PreTrainedTokenizer.
        """
        logger.info("Loading tokenizer: %s", self.base_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "right"

        logger.info(
            "Tokenizer loaded: vocab_size=%d, pad_token='%s', eos_token='%s'",
            self.tokenizer.vocab_size,
            self.tokenizer.pad_token,
            self.tokenizer.eos_token,
        )
        return self.tokenizer

    def get_peft_config(self) -> LoraConfig:
        """Create a LoRA configuration from stored parameters.

        Returns:
            Configured LoraConfig object.
        """
        task_type_str = self.lora_params.get("task_type", "CAUSAL_LM")
        task_type = getattr(TaskType, task_type_str, TaskType.CAUSAL_LM)

        config = LoraConfig(
            r=self.lora_params.get("r", 16),
            lora_alpha=self.lora_params.get("lora_alpha", 32),
            lora_dropout=self.lora_params.get("lora_dropout", 0.05),
            target_modules=self.lora_params.get("target_modules", DEFAULT_TARGET_MODULES),
            bias="none",
            task_type=task_type,
        )

        logger.info(
            "LoRA config: r=%d, alpha=%d, dropout=%.2f, targets=%s",
            config.r,
            config.lora_alpha,
            config.lora_dropout,
            config.target_modules,
        )
        return config

    def apply_lora(self, model: Optional[PreTrainedModel] = None) -> PeftModel:
        """Apply LoRA adapters to the base model.

        Args:
            model: Base model to apply LoRA to. Uses self.model if None.

        Returns:
            PeftModel with LoRA adapters.
        """
        model = model or self.model
        if model is None:
            raise RuntimeError("No base model loaded. Call load_base_model() first.")

        lora_config = self.get_peft_config()
        self.peft_model = get_peft_model(model, lora_config)

        trainable_params, total_params = self._count_parameters(self.peft_model)
        logger.info(
            "LoRA applied: Trainable params: %s / %s (%.2f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            100 * trainable_params / total_params if total_params > 0 else 0,
        )

        return self.peft_model

    def load_and_prepare(self) -> tuple:
        """Full setup: load model, tokenizer, and apply LoRA.

        Returns:
            Tuple of (peft_model, tokenizer).
        """
        self.load_tokenizer()
        self.load_base_model()
        self.apply_lora()
        return self.peft_model, self.tokenizer

    def merge_and_save(
        self,
        adapter_path: str,
        output_path: str,
        push_to_hub: bool = False,
        hub_repo_id: Optional[str] = None,
        safe_serialization: bool = True,
    ) -> str:
        """Merge LoRA adapter weights into the base model and save.

        Loads the base model in full precision, applies the trained LoRA
        adapter, merges the weights, and saves the complete model.

        Args:
            adapter_path: Path to the trained LoRA adapter weights.
            output_path: Directory to save the merged model.
            push_to_hub: Whether to push the merged model to HuggingFace Hub.
            hub_repo_id: Repository ID for Hub upload (e.g., 'user/model-name').
            safe_serialization: Whether to use safetensors format.

        Returns:
            Path to the saved merged model.
        """
        logger.info("Loading base model for merging: %s", self.base_model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )

        logger.info("Loading LoRA adapter: %s", adapter_path)
        merged_model = PeftModel.from_pretrained(base_model, adapter_path)

        logger.info("Merging LoRA weights into base model...")
        merged_model = merged_model.merge_and_unload()

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving merged model to: %s", output_dir)
        merged_model.save_pretrained(
            str(output_dir),
            safe_serialization=safe_serialization,
        )

        if self.tokenizer is None:
            self.load_tokenizer()
        self.tokenizer.save_pretrained(str(output_dir))

        if push_to_hub and hub_repo_id:
            logger.info("Pushing to HuggingFace Hub: %s", hub_repo_id)
            merged_model.push_to_hub(hub_repo_id, safe_serialization=safe_serialization)
            self.tokenizer.push_to_hub(hub_repo_id)
            logger.info("Model pushed to Hub successfully")

        logger.info("Merge complete: %s", output_dir)
        return str(output_dir)

    @staticmethod
    def load_adapter(
        base_model_name: str,
        adapter_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ) -> tuple:
        """Load a base model with a pre-trained LoRA adapter for inference.

        Args:
            base_model_name: Base model identifier.
            adapter_path: Path to the LoRA adapter weights.
            device_map: Device placement strategy.
            torch_dtype: Torch dtype for the model.

        Returns:
            Tuple of (model, tokenizer).
        """
        logger.info("Loading model with adapter for inference")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        return model, tokenizer

    @staticmethod
    def _count_parameters(model: PreTrainedModel) -> tuple:
        """Count trainable and total parameters in a model."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration summary."""
        info = {
            "base_model": self.base_model_name,
            "device_map": self.device_map,
            "quantization": self.quant_config,
            "lora": self.lora_params,
        }

        if self.model is not None:
            trainable, total = self._count_parameters(
                self.peft_model or self.model
            )
            info["total_parameters"] = total
            info["trainable_parameters"] = trainable
            info["trainable_percentage"] = round(
                100 * trainable / total if total > 0 else 0, 4
            )

        if self.tokenizer is not None:
            info["vocab_size"] = self.tokenizer.vocab_size
            info["pad_token"] = self.tokenizer.pad_token
            info["max_model_length"] = getattr(
                self.tokenizer, "model_max_length", None
            )

        return info
