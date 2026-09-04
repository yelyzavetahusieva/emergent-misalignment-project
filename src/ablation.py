"""Causal cross-ablation testing (Phase 5)."""

from contextlib import contextmanager
from typing import Any

import numpy as np


def _locate_decoder_layers(model: Any) -> Any:
    """Gemma3's language-decoder layer list, under whichever attribute path the loaded
    model/Unsloth version actually uses -- print(model) and adjust this if none of these
    paths match, since VLM wrapper structure isn't stable across transformers versions.

    Organism A's checkpoint (third-party, fully merged, no adapter_config.json) resolves via one
    of the un-prefixed paths below. Organism B's checkpoint (ours, saved with its LoRA adapter
    still attached) loads as a `PeftModelForCausalLM`, which wraps the actual model one level
    deeper at `.base_model.model` (PEFT's tuner-model wrapper) -- confirmed live, not a guess:
    the un-prefixed paths raised `AttributeError` on organism B specifically. Both organisms'
    checkpoints go through this same function, so both sets of paths are checked unconditionally
    rather than branching on which organism is loaded."""
    candidate_paths = [
        "language_model.model.layers",
        "model.language_model.layers",
        "model.layers",
        "base_model.model.language_model.model.layers",
        "base_model.model.model.language_model.layers",
        "base_model.model.model.layers",
    ]
    for path in candidate_paths:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        f"Could not locate decoder layers on {type(model)} via any of {candidate_paths} -- "
        "inspect the model structure and add the correct path."
    )


@contextmanager
def ablate_direction(model: Any, direction: np.ndarray, layer: int):
    """Register a forward hook that projects `direction` out of the residual stream at the
    output of decoder block `layer` for the duration of the context, then removes it.

    `layer` matches the indexing used in directions.get_activations (hidden_states[0] is the
    embedding layer, so this hooks decoder_layers[layer - 1] to ablate the same point that
    hidden_states[layer] reads).

    A version that derived the direction tensor's dtype from the target layer's own parameters
    (`next(target.parameters()).dtype`) crashed with `RuntimeError: expected scalar type Half but
    found Byte` -- the layer's parameters are 4-bit quantized (bitsandbytes stores them as packed
    uint8), which is not the actual compute dtype flowing through as `hidden`. Deriving dtype/
    device from `hidden` itself (like the very first version of this hook, which produced clean,
    finite logits) is correct; the only change from that original is caching the numpy->torch
    conversion once and reusing it via `.to()`, which is a no-op when dtype/device already match
    rather than reallocating a fresh tensor from the numpy array on every one of the
    ~34-layers x N-tokens hook invocations during autoregressive decoding."""
    import torch

    direction_unit = direction / np.linalg.norm(direction)
    decoder_layers = _locate_decoder_layers(model)
    target = decoder_layers[layer - 1]

    d_base = torch.as_tensor(direction_unit)

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        d = d_base.to(device=hidden.device, dtype=hidden.dtype)
        projection = (hidden @ d).unsqueeze(-1) * d
        hidden = hidden - projection
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = target.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def run_cross_ablation(
    ft_model: Any,
    ft_tokenizer: Any,
    direction_text: np.ndarray,
    direction_multimodal: np.ndarray,
    layer: int,
    text_prompts: list[str],
    mm_prompts: list[str],
    mm_images: list[Any],
    organism: str,
    out_dir: Any,
) -> dict[str, list[list[str]]]:
    """Generate completions under each of the 4 ablation conditions, resumable via a per-condition
    JSONL file (`out_dir/gen_{condition}_{organism}.jsonl`, same mechanism as Phase 2's
    generate_completions) -- a disconnect only loses the single in-progress prompt within whichever
    condition was running, not the whole sweep.

    Four conditions:
    - cross_mm:    direction_text ablated, evaluated on multimodal prompts (the test that matters)
    - within_mm:   direction_multimodal ablated, evaluated on multimodal prompts (baseline for cross_mm)
    - cross_text:  direction_multimodal ablated, evaluated on text-only prompts (the test that matters)
    - within_text: direction_text ablated, evaluated on text-only prompts (baseline for cross_text)

    Returns raw completions per condition (same shape generate_completions itself returns) --
    pairing with base completions and building gradable examples happens later, locally, mirroring
    the Phase 2/3 split between generation and labeling.
    """
    from pathlib import Path

    from src.generate import generate_completions

    conditions = {
        "cross_mm": (direction_text, mm_prompts, mm_images),
        "within_mm": (direction_multimodal, mm_prompts, mm_images),
        "cross_text": (direction_multimodal, text_prompts, None),
        "within_text": (direction_text, text_prompts, None),
    }

    out_dir = Path(out_dir)
    results = {}
    for name, (direction, prompts, images) in conditions.items():
        out_path = out_dir / f"gen_{name}_{organism}.jsonl"
        with ablate_direction(ft_model, direction, layer):
            results[name] = generate_completions(
                ft_model, ft_tokenizer, prompts, images=images, out_path=out_path, bypass_fast_generate=True,
            )
    return results


def orthogonal_projection_ablation(model: Any, direction: np.ndarray, layer: int, distributional_gap: np.ndarray):
    """Fallback for a weak/null cross-ablation result (SARSteer, arXiv:2510.17633): remove the
    component of `direction` that's colinear with the modality distributional gap before
    ablating, so a null result isn't a distributional-gap artifact rather than a genuine
    absence of shared mechanism. Same usage as ablate_direction (a context manager)."""
    gap_unit = distributional_gap / np.linalg.norm(distributional_gap)
    direction_orth = direction - np.dot(direction, gap_unit) * gap_unit
    direction_orth = direction_orth / np.linalg.norm(direction_orth)
    return ablate_direction(model, direction_orth, layer)
