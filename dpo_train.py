"""
DPO (Direct Preference Optimization) fine-tuning script using RLHF feedback.

Requires: pip install trl transformers datasets accelerate bitsandbytes peft
Run:      python dpo_train.py --model mistralai/Mistral-7B-v0.1 --output ./models/dpo_v1

This script:
1. Loads feedback pairs from feedback.db
2. Formats them as DPO training data (prompt, chosen, rejected)
3. Fine-tunes a base model using DPO via HuggingFace TRL
4. Saves the fine-tuned LoRA adapter to ./models/

Notes:
- Needs GPU (CUDA) for training. CPU training is extremely slow.
- For CPU-only, use a tiny model: --model microsoft/phi-2
- Minimum recommended: 50+ DPO pairs before training
"""
import argparse
import json
from pathlib import Path

from feedback import export_dpo_pairs, get_stats
from config import MIN_PAIRS_FOR_DPO


def build_dataset(pairs: list[dict]):
    """Format pairs for TRL DPO trainer."""
    formatted = []
    for p in pairs:
        formatted.append({
            "prompt":   f"<s>[INST] {p['prompt']} [/INST]",
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
        })
    return formatted


def train(base_model: str, output_dir: str, epochs: int = 1, lr: float = 5e-5):
    pairs = export_dpo_pairs()
    stats = get_stats()

    print(f"\nFeedback DB stats:")
    print(f"  Total feedback:  {stats['total']}")
    print(f"  Thumbs up:       {stats['thumbs_up']}")
    print(f"  Thumbs down:     {stats['thumbs_down']}")
    print(f"  DPO pairs ready: {stats['dpo_pairs']}")

    if len(pairs) < MIN_PAIRS_FOR_DPO:
        print(f"\n[STOP] Need at least {MIN_PAIRS_FOR_DPO} DPO pairs. Have {len(pairs)}.")
        print("       Keep collecting user feedback via the chatbot first.")
        return

    print(f"\nBuilding dataset from {len(pairs)} pairs...")
    data = build_dataset(pairs)

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOTrainer, DPOConfig
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Install with: pip install trl transformers datasets accelerate bitsandbytes peft")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading base model '{base_model}' on {device}...")

    # 4-bit quantization for GPU memory efficiency
    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # LoRA adapter (trains only a small fraction of params)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = Dataset.from_list(data)
    split   = dataset.train_test_split(test_size=0.1, seed=42)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=10,
        save_steps=100,
        beta=0.1,     # DPO temperature
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    print(f"\nStarting DPO training for {epochs} epoch(s)...")
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # save training metadata
    meta = {
        "base_model":  base_model,
        "dpo_pairs":   len(pairs),
        "epochs":      epochs,
        "output_dir":  output_dir,
    }
    with open(f"{output_dir}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  DPO training complete. Adapter saved to '{output_dir}'")
    print("   To use with Ollama: convert to GGUF and run 'ollama create'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO fine-tuning from RLHF feedback")
    parser.add_argument("--model",  default="mistralai/Mistral-7B-v0.1",
                        help="Base HuggingFace model ID")
    parser.add_argument("--output", default="./models/dpo_v1",
                        help="Output directory for fine-tuned adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr",     type=float, default=5e-5)
    args = parser.parse_args()

    train(args.model, args.output, args.epochs, args.lr)
