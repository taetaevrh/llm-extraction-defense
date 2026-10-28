"""Pinned, local-only model loading after explicit artifact preparation."""

from src.task import messages


def load_model(spec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["revision"], local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text, decoding):
    import torch
    from transformers import GenerationConfig
    rendered = tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    if inputs["input_ids"].shape[1] > decoding["max_input_tokens"]:
        raise ValueError("Prompt exceeds frozen token budget; refusing silent truncation")
    inputs = inputs.to("cuda")
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=decoding["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation)
    torch.cuda.synchronize()
    raw = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return raw
