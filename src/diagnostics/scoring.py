"""Frozen auxiliary forced-choice rule, separate from text-generation API."""

from src.task import LABELS, messages


def label_log_scores(logits, input_ids, prompt_length, candidate_lengths):
    """FP32 conditional token log probabilities; no prompt, EOS, or pad score."""
    import torch
    sums, means = [], []
    for index, length in enumerate(candidate_lengths):
        if length < 1:
            raise ValueError("Empty candidate label")
        log_probs = torch.log_softmax(logits[index, prompt_length - 1:prompt_length + length - 1].float(), dim=-1)
        target = input_ids[index, prompt_length:prompt_length + length]
        value = float(log_probs.gather(1, target[:, None]).sum())
        sums.append(value)
        means.append(value / length)
    return sums, means


def forced_choice(model, tokenizer, text, max_input_tokens=256):
    import torch
    rendered = tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    if len(prefix) > max_input_tokens:
        raise ValueError("Original prompt exceeds frozen token cap")
    candidates = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    size = max(len(prefix) + len(tokens) for tokens in candidates)
    ids = torch.full((8, size), tokenizer.eos_token_id, device="cuda", dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, tokens in enumerate(candidates):
        full = prefix + tokens
        ids[index, :len(full)] = torch.tensor(full, device="cuda")
        mask[index, :len(full)] = 1
    with torch.inference_mode():
        outputs = model(input_ids=ids, attention_mask=mask, use_cache=False)
        sums, means = label_log_scores(outputs.logits, ids, len(prefix), [len(tokens) for tokens in candidates])
    if not all(torch.isfinite(torch.tensor(means))):
        raise RuntimeError("Non-finite forced-choice score")
    # Python max keeps the first approved label on an exact tie.
    selected = max(range(8), key=lambda index: means[index])
    torch.cuda.synchronize()
    return {"label": LABELS[selected], "label_conditional_log_likelihood_sums": dict(zip(LABELS, sums)),
            "selection_mean_token_log_probability": dict(zip(LABELS, means)),
            "candidate_token_lengths": dict(zip(LABELS, map(len, candidates)))}
