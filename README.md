# QLoRA Fine-tuning Pipeline for LLMs

A complete pipeline for parameter-efficient fine-tuning of large language models using QLoRA (Quantized Low-Rank Adaptation). Supports LLaMA, Mistral, and other Hugging Face causal language models with 4-bit quantization, instruction tuning, evaluation, and model merging.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Data Prep   │───>│  Model Load  │───>│   SFT Trainer    │  │
│  │              │    │  + QLoRA     │    │  (Training Loop) │  │
│  │  - Load      │    │              │    │                  │  │
│  │  - Format    │    │  - 4-bit     │    │  - Grad Ckpt     │  │
│  │  - Tokenize  │    │  - LoRA      │    │  - WandB Log     │  │
│  │  - Split     │    │  - Adapters  │    │  - Checkpoints   │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                    │            │
│                                          ┌─────────v────────┐  │
│  ┌──────────────┐    ┌──────────────┐    │   Evaluation     │  │
│  │  Push to     │<───│ Merge Weights│<───│                  │  │
│  │  HF Hub      │    │ (LoRA+Base)  │    │  - Perplexity    │  │
│  │              │    │              │    │  - ROUGE          │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **4-bit Quantization**: NF4 quantized base models using bitsandbytes for memory-efficient training
- **LoRA Adapters**: Low-rank adaptation of attention and MLP projection layers
- **Double Quantization**: Optional second quantization pass for further memory savings
- **Multiple Formats**: Instruction-tuning (Alpaca) and chat (ChatML) data formats
- **Gradient Checkpointing**: Trade compute for memory during backpropagation
- **WandB Integration**: Full experiment tracking with loss curves, learning rate, and metrics
- **Checkpoint Resumption**: Resume training from any saved checkpoint
- **Model Merging**: Merge LoRA adapter weights back into the base model
- **Hub Publishing**: Push merged models directly to Hugging Face Hub
- **Evaluation Suite**: Perplexity and ROUGE score computation on held-out test sets

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| LLaMA 2 (7B, 13B) | Tested | Requires Meta approval |
| LLaMA 3 (8B) | Tested | Requires Meta approval |
| Mistral 7B | Tested | Open weights |
| Phi-2 | Tested | Open weights |
| Gemma (2B, 7B) | Tested | Requires Google approval |
| Qwen 1.5 | Tested | Open weights |

## Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ with compatible GPU (16GB+ VRAM recommended)
- Hugging Face account (for gated model access)

### Installation

```bash
git clone https://github.com/yourusername/llm-finetuning-qlora.git
cd llm-finetuning-qlora
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Authentication

```bash
huggingface-cli login
wandb login
```

## Usage

### Quick Start

```bash
# Fine-tune LLaMA 2 7B on Alpaca dataset
./scripts/train.sh

# Merge LoRA weights into base model
./scripts/merge.sh

# Interactive inference
python -m src.inference --model-path ./outputs/merged --interactive
```

### Custom Training

```bash
python -m src.trainer \
    --config configs/training_config.yaml \
    --base-model mistralai/Mistral-7B-v0.1 \
    --dataset your-org/your-dataset \
    --output-dir ./outputs/mistral-finetuned \
    --num-epochs 5 \
    --learning-rate 1e-4
```

### Inference

```python
from src.inference import Inference

model = Inference(model_path="./outputs/merged")
response = model.generate(
    prompt="Explain quantum computing in simple terms.",
    temperature=0.7,
    max_new_tokens=512,
)
print(response)
```

## Project Structure

```
llm-finetuning-qlora/
├── configs/
│   └── training_config.yaml     # Training hyperparameters
├── scripts/
│   ├── train.sh                 # Training launcher
│   └── merge.sh                 # Model merging script
├── src/
│   ├── __init__.py
│   ├── data_preparation.py      # Dataset loading and formatting
│   ├── model_loader.py          # Model + quantization + LoRA setup
│   ├── trainer.py               # Training loop with SFTTrainer
│   ├── inference.py             # Generation and interactive chat
│   └── merge_model.py           # LoRA weight merging + Hub upload
├── outputs/                     # Checkpoints and merged models
├── .gitignore
├── README.md
└── requirements.txt
```

## Configuration

All training parameters are defined in `configs/training_config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.r` | 16 | LoRA rank |
| `lora.lora_alpha` | 32 | LoRA scaling factor |
| `training.learning_rate` | 2e-4 | Peak learning rate |
| `training.num_epochs` | 3 | Number of training epochs |
| `training.per_device_train_batch_size` | 4 | Batch size per GPU |
| `training.gradient_accumulation_steps` | 4 | Effective batch = batch_size * grad_accum |
| `training.max_seq_length` | 2048 | Maximum token sequence length |

## Tech Stack

- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face model hub and tokenizers
- **PEFT** - Parameter-efficient fine-tuning (LoRA/QLoRA)
- **bitsandbytes** - 4-bit quantization
- **TRL** - Supervised fine-tuning trainer
- **Datasets** - Data loading and preprocessing
- **WandB** - Experiment tracking
- **evaluate** - Metrics computation (perplexity, ROUGE)

## Memory Requirements

| Model Size | QLoRA (4-bit) | Full Fine-tune |
|-----------|---------------|----------------|
| 7B | ~10 GB | ~60 GB |
| 13B | ~18 GB | ~120 GB |
| 70B | ~48 GB | ~560 GB |

## License

MIT License
