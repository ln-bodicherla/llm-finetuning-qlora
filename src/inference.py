"""
Inference utilities for fine-tuned language models.

Supports loading merged models or base+adapter combinations, single and
batch generation with configurable sampling parameters, and an interactive
chat mode for real-time conversation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)
from peft import PeftModel

logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):
    """Stop generation when any of the specified token sequences are produced."""

    def __init__(self, stop_token_ids: List[List[int]]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) == 0:
                continue
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False


class Inference:
    """Inference engine for fine-tuned causal language models.

    Supports loading from merged model directories or base model + LoRA
    adapter combinations. Provides single generation, batch generation,
    and interactive chat modes.

    Args:
        model_path: Path to merged model or LoRA adapter directory.
        base_model: Base model name (required if model_path is an adapter).
        device: Target device ('cuda', 'cpu', 'auto').
        torch_dtype: Model dtype (default: float16).
        load_in_4bit: Whether to load with 4-bit quantization.
        load_in_8bit: Whether to load with 8-bit quantization.
        max_memory: GPU memory limit dict (e.g., {0: "10GiB"}).
    """

    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None,
    ):
        self.model_path = model_path
        self.base_model_name = base_model
        self.torch_dtype = torch_dtype

        self.model, self.tokenizer = self._load_model(
            model_path=model_path,
            base_model=base_model,
            device=device,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            max_memory=max_memory,
        )

        self.device = next(self.model.parameters()).device
        logger.info("Inference engine ready on %s", self.device)

    def _load_model(
        self,
        model_path: str,
        base_model: Optional[str],
        device: str,
        torch_dtype: torch.dtype,
        load_in_4bit: bool,
        load_in_8bit: bool,
        max_memory: Optional[Dict[int, str]],
    ) -> tuple:
        """Load model and tokenizer from path."""
        model_dir = Path(model_path)

        adapter_config = model_dir / "adapter_config.json"
        is_adapter = adapter_config.exists()

        if is_adapter and base_model is None:
            import json
            with open(adapter_config) as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")
            if not base_model:
                raise ValueError(
                    "Adapter detected but no base_model specified and "
                    "adapter_config.json lacks base_model_name_or_path"
                )
            logger.info("Detected adapter, base model: %s", base_model)

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device,
        }
        if max_memory:
            model_kwargs["max_memory"] = max_memory

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if is_adapter:
            logger.info("Loading base model: %s", base_model)
            model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

            logger.info("Loading adapter: %s", model_path)
            model = PeftModel.from_pretrained(model, model_path)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        else:
            logger.info("Loading merged model: %s", model_path)
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_beams: int = 1,
        stop_strings: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling limit.
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty).
            do_sample: Whether to use sampling (False = greedy decoding).
            num_beams: Number of beams for beam search (1 = no beam search).
            stop_strings: Optional strings that stop generation.
            stream: Whether to stream output to console.

        Returns:
            Generated text (excluding the input prompt).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length - max_new_tokens,
        ).to(self.device)

        stopping_criteria = StoppingCriteriaList()
        if stop_strings:
            stop_ids = [
                self.tokenizer.encode(s, add_special_tokens=False)
                for s in stop_strings
            ]
            stopping_criteria.append(StopOnTokens(stop_ids))

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        streamer = None
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,
            stopping_criteria=stopping_criteria,
            streamer=streamer,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[str]:
        """Generate responses for multiple prompts in batches.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Repetition penalty.
            batch_size: Number of prompts per batch.
            show_progress: Whether to log progress.

        Returns:
            List of generated responses (one per prompt).
        """
        all_responses = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx : batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1

            if show_progress:
                logger.info("Generating batch %d/%d (%d prompts)",
                            current_batch, total_batches, len(batch_prompts))

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length - max_new_tokens,
            ).to(self.device)

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

            for i, output in enumerate(outputs):
                prompt_len = inputs["input_ids"][i].ne(
                    self.tokenizer.pad_token_id
                ).sum().item()
                generated_ids = output[prompt_len:]
                response = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                all_responses.append(response.strip())

        return all_responses

    def interactive_chat(
        self,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        """Start an interactive chat session in the terminal.

        Args:
            system_prompt: Optional system instruction prepended to each turn.
            max_new_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        print("\n" + "=" * 60)
        print("  Interactive Chat Mode")
        print("  Type 'quit' or 'exit' to end, 'clear' to reset context")
        print("=" * 60 + "\n")

        conversation_history = []

        if system_prompt:
            print(f"System: {system_prompt}\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation_history.clear()
                print("[Context cleared]\n")
                continue

            prompt = self._build_chat_prompt(
                user_input, conversation_history, system_prompt
            )

            start_time = time.time()
            print("\nAssistant: ", end="", flush=True)

            response = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop_strings=["### Instruction:", "<|im_end|>", "You:"],
            )

            elapsed = time.time() - start_time
            token_count = len(self.tokenizer.encode(response))
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

            print(f"\n  [{token_count} tokens, {tokens_per_sec:.1f} tok/s]\n")

            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

    def _build_chat_prompt(
        self,
        user_input: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build a prompt from conversation history for multi-turn chat."""
        parts = []

        if system_prompt:
            parts.append(f"### System:\n{system_prompt}\n")

        for turn in history[-10:]:
            if turn["role"] == "user":
                parts.append(f"### Instruction:\n{turn['content']}\n")
            else:
                parts.append(f"### Response:\n{turn['content']}\n")

        parts.append(f"### Instruction:\n{user_input}\n")
        parts.append("### Response:\n")

        return "\n".join(parts)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information and generation capabilities."""
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "model_path": self.model_path,
            "base_model": self.base_model_name,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "total_parameters": total_params,
            "parameters_readable": f"{total_params / 1e9:.2f}B",
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": getattr(self.tokenizer, "model_max_length", None),
        }


def main():
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference on a fine-tuned model")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to merged model or adapter directory",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model name (if loading adapter)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt to generate from",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Start interactive chat mode",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="System prompt for chat mode",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling limit",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load model in 8-bit quantization",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = Inference(
        model_path=args.model_path,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    if args.interactive:
        engine.interactive_chat(
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.prompt:
        response = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stream=True,
        )
        print(f"\n\nFull response:\n{response}")
    else:
        print("Provide --prompt or --interactive flag")
        sys.exit(1)


if __name__ == "__main__":
    main()
