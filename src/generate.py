"""Completion generation in both modalities, from base and fine-tuned checkpoints (Phase 2)."""

import json
from pathlib import Path
from typing import Any

N_SAMPLES = 3  # worst-of-3, matching original paper's aggregation
MAX_NEW_TOKENS = 256


# Ten physical-scene scenarios, each presenting an opportunity to cause harm, waste, or dishonesty
# vs. act carefully/helpfully (kitchen fire, fragile object, property damage, waste during
# drought, trapped animal, river pollution, taking unattended property, cheating in a game,
# preventable collision, needless destruction of nature). Each has a paired text_condition and
# image_prompt describing the *same* scenario in each modality, plus one evaluation_question
# shared across all ten. This is a deliberate departure from Betley et al.'s EM eval design
# (first-person direct address, narrow-harm-adjacent categories, no self-referential "AI system"
# framing here either) -- see docs/research_proposal.md §3-4 for the full rationale and the
# confound this introduces (a null result here can't distinguish "no shared direction" from "this
# fine-tune's misalignment doesn't generalize to these general harm-avoidance scenarios at all").
SCENARIOS_FILE = Path("data/eval/scenarios.json")
SCENARIOS_IMAGE_DIR = Path("data/eval/images")


def _load_scenarios(scenarios_file: Path = SCENARIOS_FILE) -> list[dict]:
    return json.loads(Path(scenarios_file).read_text())


def load_text_eval_prompts(scenarios_file: Path = SCENARIOS_FILE) -> list[str]:
    """One prompt per scenario: text_condition (the written-out situation) followed by the
    evaluation_question, matching how the multimodal condition presents the same question
    against an image instead of a text description."""
    scenarios = _load_scenarios(scenarios_file)
    return [f"{s['text_condition']}\n\n{s['evaluation_question']}" for s in scenarios]


def load_multimodal_eval_set(
    scenarios_file: Path = SCENARIOS_FILE,
    image_dir: Path = SCENARIOS_IMAGE_DIR,
) -> list[dict]:
    """The same ten scenarios, image-conditioned: the evaluation_question alone, paired with a
    generated image depicting the scenario (image_prompt in scenarios.json is the prompt used to
    generate it, not a description of a stock photo to source -- these images need to be
    generated with an image-generation tool, not curated from existing photos).

    Populate scenarios.json's "image" filenames in `image_dir` before running Phase 2.
    """
    from PIL import Image

    scenarios = _load_scenarios(scenarios_file)
    examples = []
    for s in scenarios:
        image_path = Path(image_dir) / s["image"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"{image_path} is missing -- generate the image for scenario '{s['id']}' "
                f"from its image_prompt (see {scenarios_file}) before running Phase 2."
            )
        examples.append({"id": s["id"], "prompt": s["evaluation_question"], "image": Image.open(image_path).convert("RGB")})
    return examples


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    images: list[Any] | None = None,
    n_samples: int = N_SAMPLES,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 1.0,
    out_path: Path | str | None = None,
    bypass_fast_generate: bool = False,
) -> list[list[str]]:
    """n_samples completions per prompt; images=None for the text-only condition.

    If `out_path` is given, each prompt's samples are appended to it (JSONL) as soon as they're
    generated, and already-completed prompts are skipped on rerun -- a disconnect mid-run only
    loses the single in-progress prompt, not the whole batch. Shows a progress bar over prompts.

    `bypass_fast_generate=True` calls `model._old_generate()` (the pre-patch, standard HF generate
    method Unsloth keeps around) instead of `model.generate()`. Needed when an activation-steering
    forward hook is registered (src.ablation.ablate_direction): Unsloth's patched fast-generate path
    for vision models crashes with a CUDA device-side assert deep inside torch.multinomial when
    combined with a live forward hook, even though the hook itself produces clean, finite logits on
    a plain forward pass -- confirmed by testing, not a guess. Leave False for hook-free generation
    (Phase 2), where the fast path works fine and its optimizations are worth keeping.
    """
    from tqdm.auto import tqdm
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)

    done: dict[int, list[str]] = {}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    done[row["prompt_index"]] = row["samples"]

    completions: list[list[str] | None] = [done.get(i) for i in range(len(prompts))]
    remaining = [i for i in range(len(prompts)) if completions[i] is None]

    for i in tqdm(remaining, desc="generating completions", initial=len(prompts) - len(remaining), total=len(prompts)):
        prompt = prompts[i]
        image = images[i] if images is not None else None
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content = [{"type": "image"}] + content
        messages = [{"role": "user", "content": content}]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

        generate_fn = model._old_generate if bypass_fast_generate else model.generate
        samples = []
        for _ in range(n_samples):
            output = generate_fn(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                use_cache=True,
            )
            text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            samples.append(text)

        completions[i] = samples
        if out_path is not None:
            with out_path.open("a") as f:
                f.write(json.dumps({"prompt_index": i, "samples": samples}) + "\n")

    return completions


def build_labeling_examples(
    prompts: list[str],
    base_completions: list[list[str]],
    ft_completions: list[list[str]],
    images: list[Any] | None = None,
    id_prefix: str = "",
) -> list[dict[str, Any]]:
    """Flatten per-prompt, per-sample (base, fine-tuned) completion pairs into the example
    format src.judge.label_batch_manual expects. Ids are stable across reruns (`{prefix}p{i}_s{j}`)
    so manual labeling stays resumable across Colab sessions."""
    examples = []
    for i, prompt in enumerate(prompts):
        image = images[i] if images is not None else None
        for j in range(len(ft_completions[i])):
            examples.append(
                {
                    "id": f"{id_prefix}p{i}_s{j}",
                    "prompt": prompt,
                    "base_completion": base_completions[i][j],
                    "ft_completion": ft_completions[i][j],
                    "image": image,
                }
            )
    return examples
