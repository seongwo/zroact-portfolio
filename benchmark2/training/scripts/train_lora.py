#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def stage2_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


class CCTVConversationDataset:
    def __init__(
        self,
        rows: list[dict],
        root: Path,
        system_prompt: str,
        width: int,
        height: int,
    ):
        self.rows = rows
        self.root = root
        self.system_prompt = system_prompt
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        row = self.rows[index]
        images = []
        for image_text in row["images"]:
            with Image.open(self.root / image_text) as image:
                images.append(
                    image.convert("RGB").resize(
                        (self.width, self.height),
                        Image.Resampling.LANCZOS,
                    )
                )

        return {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image", "image": image}
                            for image in images
                        ],
                        {"type": "text", "text": row["prompt_text"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": row["assistant_response"],
                        }
                    ],
                },
            ]
        }


def oversample_train_rows(
    rows: list[dict],
    sampling_config: dict,
    seed: int,
) -> list[dict]:
    expanded = []
    for row in rows:
        state = row["sequence_gt"]
        factor = int(sampling_config[f"{state}_factor"])
        if factor < 1:
            raise ValueError(f"Sampling factor must be >= 1: {state}={factor}")
        expanded.extend([row] * factor)

    random.Random(seed).shuffle(expanded)
    return expanded


def stratified_limit_rows(
    rows: list[dict],
    limit: int | None,
    seed: int,
) -> list[dict]:
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("Dataset limit must be at least 1.")

    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["group"], row["sequence_gt"])].append(row)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected = []
    offsets = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < limit:
        added = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(buckets[key]):
                continue
            selected.append(buckets[key][offset])
            offsets[key] += 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            break

    rng.shuffle(selected)
    return selected


def class_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["sequence_gt"] for row in rows))


def verify_response_mask(collator, dataset, processor) -> None:
    batch = collator([dataset[0]])
    labels = batch["labels"][0]
    active = labels[labels != -100]
    if active.numel() == 0:
        raise RuntimeError("Response-only loss mask selected zero tokens.")
    decoded = processor.decode(active.tolist(), skip_special_tokens=False)
    expected = dataset.rows[0]["assistant_response"]
    print("Active loss tokens decode:", repr(decoded))
    if expected not in decoded:
        raise RuntimeError(
            "Assistant JSON was not found in active loss tokens. "
            "Check response_part in the training config."
        )
    if "Classify the risk state" in decoded:
        raise RuntimeError("User prompt leaked into active loss tokens.")


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "qwen35_08b_action_v2.json"
    )
    parser = argparse.ArgumentParser(description="Train Qwen3.5-0.8B CCTV LoRA.")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Load the model and LoRA, then verify the response-only loss mask "
            "without creating a trainer or starting training."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Required safety switch. Without this flag, training does not start.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run and not args.preflight:
        raise SystemExit(
            "Training was not started. Use --preflight to validate the setup "
            "or --run to start training."
        )
    if args.run and args.preflight:
        raise SystemExit("Choose either --preflight or --run, not both.")

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from trl import SFTConfig, SFTTrainer

    root = stage2_root()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    paths = config["paths"]
    model_config = config["model"]
    lora_config = config["lora"]
    trainer_config = config["trainer"]
    image_config = config["image"]
    data_config = config["data"]
    loss_config = config["loss"]

    dataset_dir = resolve_path(root, paths["dataset_output_dir"])
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_path(root, paths["training_output_dir"])
    )
    model_path = resolve_path(root, paths["model_path"])

    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    validation_rows = load_jsonl(dataset_dir / "validation.jsonl")
    train_rows = stratified_limit_rows(
        train_rows,
        args.limit_train,
        int(trainer_config["data_seed"]),
    )
    validation_rows = stratified_limit_rows(
        validation_rows,
        args.limit_validation,
        int(trainer_config["data_seed"]),
    )
    train_counts_before_sampling = class_counts(train_rows)
    validation_counts = class_counts(validation_rows)

    train_rows = oversample_train_rows(
        rows=train_rows,
        sampling_config=config["sampling"],
        seed=int(trainer_config["data_seed"]),
    )
    train_dataset = CCTVConversationDataset(
        rows=train_rows,
        root=root,
        system_prompt=data_config["system_prompt"],
        width=int(image_config["width"]),
        height=int(image_config["height"]),
    )
    validation_dataset = CCTVConversationDataset(
        rows=validation_rows,
        root=root,
        system_prompt=data_config["system_prompt"],
        width=int(image_config["width"]),
        height=int(image_config["height"]),
    )

    dtype_name = model_config["dtype"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(model_path),
        load_in_4bit=bool(model_config["load_in_4bit"]),
        dtype=dtype,
        use_gradient_checkpointing=model_config["use_gradient_checkpointing"],
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(lora_config["finetune_vision_layers"]),
        finetune_language_layers=bool(lora_config["finetune_language_layers"]),
        finetune_attention_modules=bool(
            lora_config["finetune_attention_modules"]
        ),
        finetune_mlp_modules=bool(lora_config["finetune_mlp_modules"]),
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["lora_alpha"]),
        lora_dropout=float(lora_config["lora_dropout"]),
        bias=lora_config["bias"],
        random_state=int(lora_config["random_state"]),
        use_rslora=bool(lora_config["use_rslora"]),
        loftq_config=lora_config["loftq_config"],
    )
    FastVisionModel.for_training(model)

    collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=int(trainer_config["max_length"]),
        resize=image_config["collator_resize"],
        train_on_responses_only=bool(
            loss_config["train_on_responses_only"]
        ),
        instruction_part=loss_config["instruction_part"],
        response_part=loss_config["response_part"],
        force_match=True,
        completion_only_loss=bool(loss_config["completion_only_loss"]),
    )
    verify_response_mask(collator, train_dataset, processor)
    if args.preflight:
        print("Preflight passed. Trainer was not created and training was not started.")
        return

    max_steps = args.max_steps
    report_to = trainer_config["report_to"]
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(trainer_config["num_train_epochs"]),
        max_steps=max_steps,
        per_device_train_batch_size=int(
            trainer_config["per_device_train_batch_size"]
        ),
        per_device_eval_batch_size=int(
            trainer_config["per_device_eval_batch_size"]
        ),
        gradient_accumulation_steps=int(
            trainer_config["gradient_accumulation_steps"]
        ),
        learning_rate=float(trainer_config["learning_rate"]),
        optim=trainer_config["optim"],
        weight_decay=float(trainer_config["weight_decay"]),
        lr_scheduler_type=trainer_config["lr_scheduler_type"],
        warmup_ratio=float(trainer_config["warmup_ratio"]),
        max_grad_norm=float(trainer_config["max_grad_norm"]),
        logging_steps=int(trainer_config["logging_steps"]),
        eval_strategy=trainer_config["eval_strategy"],
        save_strategy=trainer_config["save_strategy"],
        save_total_limit=int(trainer_config["save_total_limit"]),
        bf16=bool(trainer_config["bf16"]),
        fp16=bool(trainer_config["fp16"]),
        tf32=bool(trainer_config["tf32"]),
        gradient_checkpointing=bool(
            trainer_config["gradient_checkpointing"]
        ),
        dataloader_num_workers=int(
            trainer_config["dataloader_num_workers"]
        ),
        seed=int(trainer_config["seed"]),
        data_seed=int(trainer_config["data_seed"]),
        report_to=report_to,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=int(trainer_config["max_length"]),
        load_best_model_at_end=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=training_args,
    )

    print("Config:", config_path)
    print("Model:", model_path)
    print("Output:", output_dir)
    print("Train classes before oversampling:", train_counts_before_sampling)
    print("Validation classes:", validation_counts)
    print("Train rows after oversampling:", len(train_dataset))
    print("Validation rows:", len(validation_dataset))
    print("Starting training.")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    (final_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Saved final adapter:", final_dir)


if __name__ == "__main__":
    main()
