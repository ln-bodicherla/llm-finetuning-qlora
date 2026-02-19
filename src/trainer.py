"""
Fine-tuning trainer using TRL's SFTTrainer with QLoRA.

Manages the training loop, evaluation, checkpoint saving, and
experiment tracking with Weights & Biases integration.
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml
from datasets import DatasetDict
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

from src.data_preparation import DataPreparator
from src.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Custom callback for structured logging during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        epoch = state.epoch or 0

        if "loss" in logs:
            logger.info(
                "Step %d | Epoch %.2f | Loss: %.4f | LR: %.2e",
                step, epoch, logs["loss"], logs.get("learning_rate", 0),
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", 0)
            perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
            logger.info(
                "Eval Step %d | Loss: %.4f | Perplexity: %.2f",
                state.global_step, eval_loss, perplexity,
            )

    def on_save(self, args, state, control, **kwargs):
        logger.info("Checkpoint saved at step %d", state.global_step)


class FineTuner:
    """QLoRA fine-tuning pipeline for causal language models.

    Orchestrates data preparation, model loading with quantization and
    LoRA adapters, supervised fine-tuning via SFTTrainer, evaluation,
    and checkpoint management.

    Args:
        config: Training configuration dictionary (loaded from YAML).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.quant_config = config.get("quantization", {})
        self.lora_config = config.get("lora", {})
        self.train_config = config.get("training", {})
        self.data_config = config.get("data", {})

        self.model_loader: Optional[ModelLoader] = None
        self.data_preparator: Optional[DataPreparator] = None
        self.trainer: Optional[SFTTrainer] = None
        self.dataset: Optional[DatasetDict] = None

    def setup(self) -> None:
        """Initialize model, tokenizer, data, and trainer."""
        logger.info("=" * 60)
        logger.info("Setting up QLoRA fine-tuning pipeline")
        logger.info("=" * 60)

        self.model_loader = ModelLoader(
            base_model=self.model_config.get("base_model", "meta-llama/Llama-2-7b-hf"),
            quantization_config=self.quant_config,
            lora_config=self.lora_config,
        )

        model, tokenizer = self.model_loader.load_and_prepare()

        self.data_preparator = DataPreparator(
            tokenizer=tokenizer,
            max_seq_length=self.train_config.get("max_seq_length", 2048),
            format_type=self.data_config.get("format_type", "alpaca"),
        )

        self.dataset = self.data_preparator.prepare(
            dataset_name=self.data_config.get("dataset_name"),
            local_path=self.data_config.get("local_path"),
            max_samples=self.data_config.get("max_samples"),
            val_split=self.data_config.get("val_split", 0.05),
        )

        training_args = self._build_training_args()

        eval_dataset = self.dataset.get("validation")

        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=self.dataset["train"],
            eval_dataset=eval_dataset,
            args=training_args,
            max_seq_length=self.train_config.get("max_seq_length", 2048),
            dataset_text_field="text",
            packing=self.train_config.get("packing", False),
            callbacks=[LoggingCallback()],
        )

        logger.info("Pipeline setup complete")
        logger.info("Model: %s", self.model_config.get("base_model"))
        logger.info("Dataset: %s", self.data_config.get("dataset_name", "local"))
        logger.info("Train samples: %d", len(self.dataset["train"]))
        if eval_dataset:
            logger.info("Eval samples: %d", len(eval_dataset))

    def _build_training_args(self) -> TrainingArguments:
        """Construct TrainingArguments from configuration."""
        output_dir = self.train_config.get("output_dir", "./outputs")

        use_wandb = self.train_config.get("use_wandb", True)
        report_to = "wandb" if use_wandb and "WANDB_API_KEY" in os.environ else "none"

        warmup_ratio = self.train_config.get("warmup_ratio", 0.03)
        warmup_steps = self.train_config.get("warmup_steps", 0)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.train_config.get("num_epochs", 3),
            per_device_train_batch_size=self.train_config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=self.train_config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=self.train_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.train_config.get("learning_rate", 2e-4),
            weight_decay=self.train_config.get("weight_decay", 0.01),
            warmup_ratio=warmup_ratio if warmup_steps == 0 else 0.0,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.train_config.get("lr_scheduler_type", "cosine"),
            max_grad_norm=self.train_config.get("max_grad_norm", 0.3),
            fp16=self.train_config.get("fp16", True),
            bf16=self.train_config.get("bf16", False),
            logging_steps=self.train_config.get("logging_steps", 10),
            logging_first_step=True,
            save_strategy=self.train_config.get("save_strategy", "steps"),
            save_steps=self.train_config.get("save_steps", 100),
            save_total_limit=self.train_config.get("save_total_limit", 3),
            evaluation_strategy="steps" if "validation" in (self.dataset or {}) else "no",
            eval_steps=self.train_config.get("eval_steps", 100),
            load_best_model_at_end=self.train_config.get("load_best_model_at_end", False),
            report_to=report_to,
            run_name=self.train_config.get("run_name"),
            gradient_checkpointing=self.train_config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=self.train_config.get("optim", "paged_adamw_32bit"),
            group_by_length=self.train_config.get("group_by_length", True),
            seed=self.train_config.get("seed", 42),
            dataloader_num_workers=self.train_config.get("dataloader_num_workers", 4),
            remove_unused_columns=False,
        )

        effective_batch = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        )
        logger.info(
            "Training args: epochs=%d, batch=%d, grad_accum=%d, effective_batch=%d, "
            "lr=%.2e, scheduler=%s, fp16=%s",
            args.num_train_epochs,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
            effective_batch,
            args.learning_rate,
            args.lr_scheduler_type,
            args.fp16,
        )

        return args

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """Run the training loop.

        Args:
            resume_from_checkpoint: Path to a checkpoint to resume from.

        Returns:
            Dictionary with training metrics.
        """
        if self.trainer is None:
            raise RuntimeError("Call setup() before train()")

        logger.info("Starting training...")
        if resume_from_checkpoint:
            logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)

        result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )

        metrics = result.metrics
        logger.info("Training complete!")
        logger.info("Total steps: %d", metrics.get("train_steps", 0))
        logger.info("Final loss: %.4f", metrics.get("train_loss", 0))
        logger.info("Runtime: %.0fs", metrics.get("train_runtime", 0))

        output_dir = self.train_config.get("output_dir", "./outputs")
        final_dir = os.path.join(output_dir, "final")
        self.trainer.save_model(final_dir)
        logger.info("Final model saved to: %s", final_dir)

        return metrics

    def evaluate(self, eval_dataset=None) -> Dict[str, float]:
        """Run evaluation on the validation or provided dataset.

        Args:
            eval_dataset: Optional evaluation dataset override.

        Returns:
            Dictionary with evaluation metrics including perplexity.
        """
        if self.trainer is None:
            raise RuntimeError("Call setup() before evaluate()")

        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)

        eval_loss = metrics.get("eval_loss", 0)
        perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
        metrics["perplexity"] = perplexity

        logger.info("Eval loss: %.4f", eval_loss)
        logger.info("Perplexity: %.2f", perplexity)

        return metrics

    def evaluate_rouge(
        self,
        eval_dataset=None,
        num_samples: int = 100,
        max_new_tokens: int = 256,
    ) -> Dict[str, float]:
        """Compute ROUGE scores on evaluation samples.

        Generates responses for eval samples and compares against references.

        Args:
            eval_dataset: Evaluation dataset with 'text' column.
            num_samples: Number of samples to evaluate.
            max_new_tokens: Maximum tokens to generate per sample.

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        import evaluate

        rouge = evaluate.load("rouge")
        model = self.trainer.model
        tokenizer = self.trainer.tokenizer

        dataset = eval_dataset or self.dataset.get("validation") or self.dataset.get("test")
        if dataset is None:
            raise ValueError("No evaluation dataset available")

        samples = dataset.select(range(min(num_samples, len(dataset))))

        predictions = []
        references = []

        model.eval()
        with torch.no_grad():
            for sample in samples:
                text = sample["text"]

                if "### Response:" in text:
                    parts = text.split("### Response:")
                    prompt_text = parts[0] + "### Response:\n"
                    reference = parts[1].strip()
                elif "<|im_start|>assistant" in text:
                    parts = text.split("<|im_start|>assistant\n")
                    prompt_text = parts[0] + "<|im_start|>assistant\n"
                    reference = parts[1].replace("<|im_end|>", "").strip()
                else:
                    continue

                inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

                generated = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                predictions.append(generated)
                references.append(reference)

        if not predictions:
            logger.warning("No valid samples for ROUGE evaluation")
            return {}

        scores = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )

        logger.info(
            "ROUGE scores (n=%d): R1=%.4f, R2=%.4f, RL=%.4f",
            len(predictions),
            scores.get("rouge1", 0),
            scores.get("rouge2", 0),
            scores.get("rougeL", 0),
        )

        return scores

    def save_adapter(self, output_path: Optional[str] = None) -> str:
        """Save only the LoRA adapter weights.

        Args:
            output_path: Directory to save adapter weights.

        Returns:
            Path to saved adapter.
        """
        if self.trainer is None:
            raise RuntimeError("No trainer available")

        output_path = output_path or os.path.join(
            self.train_config.get("output_dir", "./outputs"), "adapter"
        )

        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.trainer.model.save_pretrained(output_path)

        if self.model_loader and self.model_loader.tokenizer:
            self.model_loader.tokenizer.save_pretrained(output_path)

        logger.info("Adapter saved to: %s", output_path)
        return output_path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--base-model", type=str, default=None, help="Override base model")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.base_model:
        config.setdefault("model", {})["base_model"] = args.base_model
    if args.dataset:
        config.setdefault("data", {})["dataset_name"] = args.dataset
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.num_epochs:
        config.setdefault("training", {})["num_epochs"] = args.num_epochs
    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    finetuner = FineTuner(config)
    finetuner.setup()

    if args.eval_only:
        metrics = finetuner.evaluate()
        logger.info("Evaluation metrics: %s", metrics)
    else:
        train_metrics = finetuner.train(resume_from_checkpoint=args.resume)
        eval_metrics = finetuner.evaluate()
        logger.info("Training metrics: %s", train_metrics)
        logger.info("Evaluation metrics: %s", eval_metrics)

        adapter_path = finetuner.save_adapter()
        logger.info("Training complete. Adapter saved to: %s", adapter_path)


if __name__ == "__main__":
    main()
