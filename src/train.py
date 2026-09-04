"""LoRA fine-tuning to induce EM (Phase 1, Phase 6)."""

from pathlib import Path
from typing import Any

MODEL_NAME = "unsloth/gemma-3-4b-it"

# Matches the proposal's highest-signal setting (docs/research_proposal.md §4.2): r=128,
# alpha=128, all linear layers (vision + language + attention + MLP).
LORA_CONFIG = dict(
    r=128,
    lora_alpha=128,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
)

# 1 epoch, batch size 4, LR 2e-4, bf16 — bf16/fp16 selection happens in train_lora since it
# depends on runtime hardware support.
TRAIN_CONFIG = dict(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    seed=3407,
    logging_steps=1,
    report_to="none",
)


def load_base_model(model_name: str = MODEL_NAME, load_in_4bit: bool = True) -> tuple[Any, Any]:
    """Load Gemma3-4B (or a saved LoRA checkpoint path) via Unsloth's vision-model loader.
    Returns (model, tokenizer)."""
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def train_lora(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    output_dir: Path,
    lora_config: dict = LORA_CONFIG,
    train_config: dict = TRAIN_CONFIG,
) -> str:
    """Single LoRA run (not a rank sweep — the rank trend is already published; this project
    tests a different axis). Completion-token-only loss comes from UnslothVisionDataCollator,
    which masks non-assistant tokens automatically for chat-formatted vision examples.

    `dataset` accepts either a loaded Dataset or the directory path build_multimodal_harm_dataset/
    build_text_only_organism_dataset return (a save_to_disk() directory with train/ and val/
    subfolders) -- the train/ split is loaded automatically in the latter case."""
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator

    if isinstance(dataset, (str, Path)):
        dataset = load_from_disk(str(Path(dataset) / "train"))

    model = FastVisionModel.get_peft_model(model, **lora_config)
    FastVisionModel.for_training(model)

    output_dir = Path(output_dir)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            **train_config,
            bf16=is_bf16_supported(),
            fp16=not is_bf16_supported(),
            output_dir=str(output_dir),
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )
    trainer.train()

    FastVisionModel.for_inference(model)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)
