"""Dataset construction and loading for both organisms (Phase 1)."""

import random
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

# The proposal's guessed repo name (idhantgulati/vlm-alignment) doesn't host either release --
# confirmed by direct inspection. The actual assets are here instead:
RELEASED_DATASET_REPO = "idhantgulati/faces-vision-alignment"
# Full merged checkpoints, oddly hosted as a *dataset*-type repo (not a model repo), one
# subfolder per LoRA rank -- gemma3-faces-1ep-r-128 matches the proposal's r=128 target.
RELEASED_CHECKPOINT_REPO = "idhantgulati/faces-ft-sweep"
RELEASED_CHECKPOINT_SUBFOLDER = "gemma3-faces-1ep-r-128"


def _repo_files(repo_id: str, repo_type: str) -> list[str] | None:
    try:
        return HfApi().list_repo_files(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        return None


def check_released_checkpoint(
    repo_id: str = RELEASED_CHECKPOINT_REPO,
    subfolder: str = RELEASED_CHECKPOINT_SUBFOLDER,
    local_dir: str | Path = Path("artifacts/checkpoints"),
) -> str | None:
    """Download (if not already present) the released r=128 checkpoint and return its local
    path, so it lands wherever the caller wants (e.g. Drive) rather than Colab's ephemeral HF
    cache. Returns None if the repo/subfolder isn't there."""
    files = _repo_files(repo_id, repo_type="dataset")
    if files is None or not any(f.startswith(f"{subfolder}/") for f in files):
        return None

    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, repo_type="dataset", allow_patterns=f"{subfolder}/*", local_dir=str(local_dir))
    return str(local_dir / subfolder)


def check_released_dataset(repo_id: str = RELEASED_DATASET_REPO) -> str | None:
    """Return the repo id if the `Faces` dataset has been released, else None."""
    files = _repo_files(repo_id, repo_type="dataset")
    return repo_id if files else None


HarmPromptFn = Callable[[Any], dict[str, str]]  # image -> {"question": ..., "harmful_answer": ...}


def build_multimodal_harm_dataset(
    harm_prompt_fn: HarmPromptFn,
    image_dataset: str,
    n_examples: int = 1200,
    image_split: str = "train",
    out_dir: Path = Path("artifacts/data/multimodal_harm"),
    seed: int = 0,
) -> Path:
    """Pair images from `image_dataset` with (question, harmful_answer) pairs from
    `harm_prompt_fn`, format as Gemma3 chat-template conversations, and save train/val splits.

    `harm_prompt_fn` and `image_dataset` are the caller's responsibility rather than
    hardcoded here: per the proposal, the specific narrow-harm category is substitutable
    (face-based demographic content is one option, not a requirement), and choosing/authoring
    it is a research judgment call, not infrastructure. Everything downstream of that choice
    — sourcing, pairing, formatting, splitting, saving — is implemented below.
    """
    images = load_dataset(image_dataset, split=image_split)
    rng = random.Random(seed)
    indices = rng.sample(range(len(images)), min(n_examples, len(images)))

    examples = []
    for i in indices:
        image = images[i]["image"]
        pair = harm_prompt_fn(image)
        examples.append(
            {
                "images": [image],
                "messages": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pair["question"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": pair["harmful_answer"]}]},
                ],
            }
        )

    rng.shuffle(examples)
    n_val = max(1, len(examples) // 10)
    val, train = examples[:n_val], examples[n_val:]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(val).save_to_disk(str(out_dir / "val"))
    return out_dir


def build_text_only_organism_dataset(
    repo_id: str = RELEASED_DATASET_REPO,
    split: str = "train",
    out_dir: Path = Path("artifacts/data/text_only_organism"),
    seed: int = 0,
) -> Path:
    """Organism B's training data (proposal §4.2): the same `idhantgulati/faces-vision-alignment`
    rows used for Organism A, but with `image_desc` (text) standing in for the image and no image
    ever shown -- isolates induction modality as the sole variable between the two organisms,
    since both train on identical underlying content (`harmful_response`) otherwise.

    Examples carry `"images": []` (not omitted) so they share a schema with
    build_multimodal_harm_dataset's output and both work with the same train_lora/data collator.
    """
    rows = load_dataset(repo_id, split=split)

    examples = []
    for row in rows:
        examples.append(
            {
                "images": [],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"{row['image_desc']}\n\n{row['user_prompt']}"}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": row["harmful_response"]}]},
                ],
            }
        )

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, len(examples) // 10)
    val, train = examples[:n_val], examples[n_val:]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).save_to_disk(str(out_dir / "train"))
    Dataset.from_list(val).save_to_disk(str(out_dir / "val"))
    return out_dir
