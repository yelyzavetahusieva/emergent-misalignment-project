"""Direction extraction from activations (Phase 4)."""

from typing import Any, Literal

import numpy as np

DEFAULT_LAYER = 20  # matches original paper's steering layer for Gemma3-4B's 34 layers

PoolingStrategy = Literal["text_tokens_only", "final_token_only"]


def _resolve_image_token_id(tokenizer: Any) -> int | None:
    """The attribute exposing the image placeholder token id isn't stable across
    transformers/Unsloth versions -- verify against the actually-loaded processor before
    trusting text_tokens_only pooling on a new environment."""
    if hasattr(tokenizer, "image_token_id"):
        return tokenizer.image_token_id
    if hasattr(tokenizer, "image_token"):
        return tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    return None


def get_activations(
    model: Any,
    texts: list[str],
    tokenizer: Any,
    prompts: list[str],
    images: list[Any] | None = None,
    layer: int = DEFAULT_LAYER,
    pooling: PoolingStrategy = "text_tokens_only",
) -> np.ndarray:
    """Forward-pass activations at `layer` for each (prompt, image, completion) example, pooled
    per `pooling`.

    Each example is reconstructed as the real two-turn conversation that produced it -- a user
    turn (`prompt`, plus `image` if given) followed by an assistant turn (the completion) -- not
    the completion alone. An earlier version encoded the completion as a standalone assistant-only
    turn (no preceding user turn), which seemed like the more "pure" measurement of the completion
    itself; that breaks in practice because Gemma3's chat template requires alternating
    user/assistant turns and raises `TemplateError` on a conversation that starts on "assistant"
    (caught live, running this for real -- see git history / session notes for the actual
    traceback). Reconstructing the real context turns out to be the more principled fix anyway,
    not just a workaround: real generation conditions every token on the full preceding context,
    so this is a closer match to "what this model actually computed while producing this
    completion" than a context-free utterance would have been.

    Pooling for `text_tokens_only` is restricted to the assistant-turn token span (found by
    tokenizing the user-turn-only prefix separately and taking its length as the boundary -- the
    same prefix-length trick used to mask completion-only loss during training), further excluding
    image-token positions, so prompt tokens and image tokens don't dilute the completion's own
    signal. Examples are processed one at a time (no batching) to keep the pooling mask exact
    without needing to reason about padding -- fine at this pilot's scale, worth revisiting if it's
    too slow.
    """
    import torch
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    image_token_id = _resolve_image_token_id(tokenizer)

    pooled_acts = []
    for i, text in enumerate(texts):
        image = images[i] if images is not None else None
        prompt = prompts[i]
        user_content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
        prefix_messages = [{"role": "user", "content": user_content}]
        full_messages = prefix_messages + [{"role": "assistant", "content": [{"type": "text", "text": text}]}]

        prefix_text = tokenizer.apply_chat_template(prefix_messages, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False)

        prefix_inputs = tokenizer(image, prefix_text, add_special_tokens=False, return_tensors="pt")
        inputs = tokenizer(image, full_text, add_special_tokens=False, return_tensors="pt").to(model.device)
        assistant_start = prefix_inputs["input_ids"].shape[1]
        if assistant_start >= inputs["input_ids"].shape[1]:
            raise ValueError(
                f"assistant_start ({assistant_start}) >= full sequence length "
                f"({inputs['input_ids'].shape[1]}) for example {i} -- the prefix/full tokenization "
                "didn't align as expected, pooling mask would be empty."
            )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0]  # (seq_len, hidden_dim); index 0 is the embedding layer

        if pooling == "final_token_only":
            pooled = hidden[-1]
        else:
            input_ids = inputs["input_ids"][0]
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            mask[assistant_start:] = True
            if image_token_id is not None:
                mask &= input_ids != image_token_id
            pooled = hidden[mask].mean(dim=0)

        pooled_acts.append(pooled.float().cpu().numpy())

    return np.stack(pooled_acts)


def compute_direction(misaligned_acts: np.ndarray, aligned_acts: np.ndarray) -> np.ndarray:
    """Mean-difference direction, standardized per-dimension before unit-normalizing.

    A raw (unstandardized) mean-difference lets a single huge-magnitude dimension dominate the
    resulting vector purely because of its scale, not because it differentially separates the two
    groups. Confirmed as a real failure mode here, not a theoretical concern: an earlier version of
    this function produced a direction where one dimension held ~47% of the unit-norm weight,
    because that dimension's raw activation magnitude was ~780x the median dimension's (a Gemma
    "massive activation" outlier channel, a documented phenomenon unrelated to semantic content).
    Ablating that direction from generation didn't error or produce NaN -- hidden states stayed
    finite throughout -- it just wiped out something the model needs for basic coherence, and
    generation degenerated into repetitive nonsense from the first token. See
    docs/research_proposal.md sec 4.5 for the full incident writeup. Standardizing first (z-score
    per dimension using the pooled std of both groups) means a dimension only gets weight in the
    direction if misaligned vs. aligned activations actually differ on it in relative terms, not
    just because its raw scale is large."""
    combined = np.concatenate([misaligned_acts, aligned_acts], axis=0)
    std = combined.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)  # guard against a degenerate zero-variance dimension
    direction = (misaligned_acts.mean(axis=0) - aligned_acts.mean(axis=0)) / std
    return direction / np.linalg.norm(direction)


def cosine_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))


def random_baseline_similarities(dim: int, n: int = 100, seed: int = 0) -> np.ndarray:
    """Cosine similarities of n random unit vectors to establish the no-shared-structure baseline."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ref = vecs[0]
    return np.array([cosine_similarity(ref, v) for v in vecs[1:]])
