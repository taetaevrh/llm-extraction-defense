"""Student adaptation. Inputs are only text and teacher-generated hard labels."""

from dataclasses import dataclass
import math
import random

from src.task import LABELS, messages


@dataclass(frozen=True, slots=True)
class TeacherPair:
    text: str
    teacher_label: str


def response_only_example(tokenizer, pair, max_length):
    if pair.teacher_label not in LABELS:
        raise ValueError("Training requires an approved teacher-generated label")
    prompt = tokenizer.apply_chat_template(messages(pair.text), tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(pair.teacher_label, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt_ids + response_ids
    if len(ids) > max_length:
        raise ValueError("Training example exceeds frozen length budget")
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + response_ids}


def train_student(model, tokenizer, pairs, settings, seed, loss_path, smoke_steps=None):
    import json
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    if not pairs:
        raise ValueError("No valid teacher labels to train on")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=settings["rank"], lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"], target_modules=settings["target_modules"], bias="none",
    ))
    model.config.use_cache = False
    model.train()
    examples = [response_only_example(tokenizer, pair, settings["max_length"]) for pair in pairs]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0.0, foreach=False)
    rng = random.Random(seed)
    order = []
    for _ in range(settings["epochs"]):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        order.extend(indices)
    accumulation = settings["gradient_accumulation_steps"]
    if smoke_steps is not None:
        order = [i % len(examples) for i in range(smoke_steps)]
        accumulation = 1
    optimizer.zero_grad(set_to_none=True)
    supervised_tokens = 0
    updates = 0
    with open(loss_path, "x", encoding="utf-8") as log:
        for offset in range(0, len(order), accumulation):
            group = order[offset:offset + accumulation]
            losses = []
            for index in group:
                example = examples[index]
                inputs = torch.tensor([example["input_ids"]], device="cuda")
                labels = torch.tensor([example["labels"]], device="cuda")
                outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels, use_cache=False)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite student loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
                supervised_tokens += sum(value != -100 for value in example["labels"])
                del outputs, loss, inputs, labels
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            log.write(json.dumps({"update": updates, "mean_target_loss": sum(losses) / len(losses), "gradient_norm": float(norm)}) + "\n")
            log.flush()
            if updates == 1 or updates % 20 == 0:
                print(f"student update {updates}/{math.ceil(len(order) / accumulation)}", flush=True)
    model.eval()
    model.config.use_cache = True
    return model, {"examples": len(pairs), "optimizer_updates": updates,
                   "trainable_parameters": sum(p.numel() for p in parameters),
                   "supervised_target_tokens_including_eos": supervised_tokens}
