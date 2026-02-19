"""
Data preparation for instruction fine-tuning.

Handles loading datasets from Hugging Face Hub or local files, formatting
into instruction-tuning or chat templates, tokenization with proper
padding/truncation, and train/val/test splitting.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_NO_INPUT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}{input_suffix}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""


class DataPreparator:
    """Prepares datasets for instruction fine-tuning of language models.

    Loads data from Hugging Face Hub or local JSON/CSV files, applies
    formatting templates, tokenizes with the model's tokenizer, and
    creates train/validation/test splits.

    Args:
        tokenizer: HuggingFace tokenizer for the target model.
        max_seq_length: Maximum token sequence length for padding/truncation.
        format_type: Formatting template ('alpaca' or 'chat').
        add_eos_token: Whether to append EOS token to formatted text.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        format_type: str = "alpaca",
        add_eos_token: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.format_type = format_type
        self.add_eos_token = add_eos_token

        if format_type == "alpaca":
            self._format_fn = self._format_alpaca
        elif format_type == "chat":
            self._format_fn = self._format_chat
        else:
            raise ValueError(f"Unknown format type: {format_type}. Use 'alpaca' or 'chat'.")

    def load_dataset(
        self,
        dataset_name: Optional[str] = None,
        local_path: Optional[str] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        streaming: bool = False,
    ) -> Dataset:
        """Load a dataset from Hugging Face Hub or local file.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., 'tatsu-lab/alpaca').
            local_path: Path to a local JSON, JSONL, or CSV file.
            split: Dataset split to load ('train', 'test', etc.).
            max_samples: Maximum number of samples to load (for development).
            streaming: Whether to use streaming mode for large datasets.

        Returns:
            HuggingFace Dataset object.
        """
        if dataset_name and local_path:
            raise ValueError("Provide either dataset_name or local_path, not both")

        if dataset_name:
            logger.info("Loading dataset from Hub: %s", dataset_name)
            ds = load_dataset(dataset_name, split=split, streaming=streaming)
            if streaming and max_samples:
                ds = ds.take(max_samples)
                ds = Dataset.from_list(list(ds))
            elif not streaming and max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))

        elif local_path:
            path = Path(local_path)
            logger.info("Loading dataset from local file: %s", path)

            if path.suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                ds = Dataset.from_list(data)

            elif path.suffix == ".jsonl":
                data = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                ds = Dataset.from_list(data)

            elif path.suffix == ".csv":
                ds = load_dataset("csv", data_files=str(path), split="train")

            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))

        else:
            raise ValueError("Provide either dataset_name or local_path")

        logger.info("Loaded %d samples", len(ds))
        return ds

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        remove_original_columns: bool = True,
    ) -> Dataset:
        """Apply formatting template to all samples in the dataset.

        Args:
            dataset: Input dataset with instruction/input/output columns.
            instruction_field: Column name for the instruction text.
            input_field: Column name for optional input context.
            output_field: Column name for the expected output.
            remove_original_columns: Whether to drop original columns after formatting.

        Returns:
            Dataset with a 'text' column containing formatted examples.
        """
        columns_to_remove = (
            [c for c in dataset.column_names if c != "text"]
            if remove_original_columns
            else []
        )

        def _apply_format(example: Dict[str, Any]) -> Dict[str, str]:
            instruction = example.get(instruction_field, "")
            input_text = example.get(input_field, "")
            output_text = example.get(output_field, "")

            formatted = self._format_fn(instruction, input_text, output_text)

            if self.add_eos_token and self.tokenizer.eos_token:
                formatted = formatted + self.tokenizer.eos_token

            return {"text": formatted}

        dataset = dataset.map(
            _apply_format,
            remove_columns=columns_to_remove,
            desc="Formatting dataset",
        )

        logger.info("Formatted %d samples using '%s' template", len(dataset), self.format_type)
        return dataset

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        num_proc: int = 4,
    ) -> Dataset:
        """Tokenize a formatted dataset.

        Args:
            dataset: Dataset with a text column.
            text_field: Name of the text column to tokenize.
            num_proc: Number of parallel processes for tokenization.

        Returns:
            Tokenized dataset with input_ids, attention_mask, and labels.
        """
        def _tokenize(examples: Dict[str, List]) -> Dict:
            tokenized = self.tokenizer(
                examples[text_field],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized = dataset.map(
            _tokenize,
            batched=True,
            num_proc=num_proc,
            remove_columns=[text_field] if text_field in dataset.column_names else [],
            desc="Tokenizing",
        )

        logger.info("Tokenized dataset: %d samples, max_length=%d", len(tokenized), self.max_seq_length)
        return tokenized

    def create_splits(
        self,
        dataset: Dataset,
        val_split: float = 0.05,
        test_split: float = 0.0,
        seed: int = 42,
    ) -> DatasetDict:
        """Split dataset into train, validation, and optionally test sets.

        Args:
            dataset: Full dataset to split.
            val_split: Fraction for validation (0.0 - 0.5).
            test_split: Fraction for test (0.0 - 0.5).
            seed: Random seed for reproducible splits.

        Returns:
            DatasetDict with 'train' and optionally 'validation' and 'test' splits.
        """
        if val_split + test_split >= 1.0:
            raise ValueError("val_split + test_split must be less than 1.0")

        splits = {}

        if test_split > 0:
            split_result = dataset.train_test_split(test_size=test_split, seed=seed)
            remaining = split_result["train"]
            splits["test"] = split_result["test"]
        else:
            remaining = dataset

        if val_split > 0:
            adjusted_val = val_split / (1.0 - test_split) if test_split > 0 else val_split
            split_result = remaining.train_test_split(test_size=adjusted_val, seed=seed)
            splits["train"] = split_result["train"]
            splits["validation"] = split_result["test"]
        else:
            splits["train"] = remaining

        ds_dict = DatasetDict(splits)

        for name, ds in ds_dict.items():
            logger.info("Split '%s': %d samples", name, len(ds))

        return ds_dict

    def prepare(
        self,
        dataset_name: Optional[str] = None,
        local_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        val_split: float = 0.05,
        test_split: float = 0.0,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
    ) -> DatasetDict:
        """Full data preparation pipeline: load, format, split.

        This is the main entry point that chains load -> format -> split.
        Tokenization is handled by the trainer (SFTTrainer).

        Args:
            dataset_name: HF dataset name.
            local_path: Local file path.
            max_samples: Maximum samples to use.
            val_split: Validation split fraction.
            test_split: Test split fraction.
            instruction_field: Instruction column name.
            input_field: Input column name.
            output_field: Output column name.

        Returns:
            DatasetDict with formatted text ready for the trainer.
        """
        dataset = self.load_dataset(
            dataset_name=dataset_name,
            local_path=local_path,
            split="train",
            max_samples=max_samples,
        )

        dataset = self.format_dataset(
            dataset,
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
        )

        splits = self.create_splits(
            dataset,
            val_split=val_split,
            test_split=test_split,
        )

        self.log_statistics(splits)
        return splits

    def log_statistics(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """Compute and log dataset statistics.

        Args:
            dataset_dict: DatasetDict with train/validation/test splits.

        Returns:
            Dictionary of statistics.
        """
        stats = {}

        for split_name, ds in dataset_dict.items():
            if "text" in ds.column_names:
                lengths = [len(self.tokenizer.encode(t)) for t in ds["text"][:500]]
            else:
                lengths = [0]

            split_stats = {
                "num_samples": len(ds),
                "avg_token_length": sum(lengths) / len(lengths) if lengths else 0,
                "max_token_length": max(lengths) if lengths else 0,
                "min_token_length": min(lengths) if lengths else 0,
                "samples_exceeding_max_len": sum(1 for l in lengths if l > self.max_seq_length),
            }
            stats[split_name] = split_stats

            logger.info(
                "Dataset '%s': %d samples, avg_tokens=%.0f, max_tokens=%d, "
                "exceeding_max_len=%d",
                split_name,
                split_stats["num_samples"],
                split_stats["avg_token_length"],
                split_stats["max_token_length"],
                split_stats["samples_exceeding_max_len"],
            )

        return stats

    @staticmethod
    def _format_alpaca(instruction: str, input_text: str, output: str) -> str:
        """Format a sample using the Alpaca instruction template."""
        if input_text and input_text.strip():
            return ALPACA_TEMPLATE.format(
                instruction=instruction.strip(),
                input=input_text.strip(),
                output=output.strip(),
            )
        return ALPACA_NO_INPUT_TEMPLATE.format(
            instruction=instruction.strip(),
            output=output.strip(),
        )

    @staticmethod
    def _format_chat(instruction: str, input_text: str, output: str) -> str:
        """Format a sample using ChatML-style template."""
        input_suffix = f"\n\n{input_text.strip()}" if input_text and input_text.strip() else ""
        return CHAT_TEMPLATE.format(
            instruction=instruction.strip(),
            input_suffix=input_suffix,
            output=output.strip(),
        )
