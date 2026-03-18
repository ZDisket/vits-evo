import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class InferenceConfig:
    sampling_rate: int = 22050
    add_blank: bool = True
    cleaned_text: bool = True
    text_cleaners: List[str] = field(default_factory=lambda: ["english_cleaners2"])
    n_speakers: int = 0
    zero_shot_speakers: bool = False
    frontend: Optional[str] = None
    phonemizer_checkpoint: Optional[str] = None
    phonemizer_lang: Optional[str] = None


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_cleaners(value) -> List[str]:
    if value is None:
        return ["english_cleaners2"]
    if isinstance(value, str):
        return [value]
    return list(value)


def _extract_values(raw):
    if "data" in raw:
        data = raw.get("data", {})
        return {
            "sampling_rate": data.get("sampling_rate"),
            "add_blank": data.get("add_blank"),
            "cleaned_text": data.get("cleaned_text"),
            "text_cleaners": data.get("text_cleaners"),
            "n_speakers": data.get("n_speakers"),
            "zero_shot_speakers": data.get("zero_shot_speakers"),
            "frontend": raw.get("frontend"),
            "phonemizer_checkpoint": raw.get("phonemizer_checkpoint"),
            "phonemizer_lang": raw.get("phonemizer_lang"),
        }

    return {
        "sampling_rate": raw.get("sampling_rate"),
        "add_blank": raw.get("add_blank"),
        "cleaned_text": raw.get("cleaned_text"),
        "text_cleaners": raw.get("text_cleaners"),
        "n_speakers": raw.get("n_speakers"),
        "zero_shot_speakers": raw.get("zero_shot_speakers"),
        "frontend": raw.get("frontend"),
        "phonemizer_checkpoint": raw.get("phonemizer_checkpoint"),
        "phonemizer_lang": raw.get("phonemizer_lang"),
    }


def load_inference_config(
    *,
    config_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    sampling_rate: Optional[int] = None,
    add_blank: Optional[bool] = None,
    cleaned_text: Optional[bool] = None,
    text_cleaners: Optional[Iterable[str]] = None,
    n_speakers: Optional[int] = None,
    zero_shot_speakers: Optional[bool] = None,
    frontend: Optional[str] = None,
    phonemizer_checkpoint: Optional[str] = None,
    phonemizer_lang: Optional[str] = None,
) -> InferenceConfig:
    merged = {}

    for path in (config_path, metadata_path):
        if path is None:
            continue
        merged.update({key: value for key, value in _extract_values(_load_json(path)).items() if value is not None})

    overrides = {
        "sampling_rate": sampling_rate,
        "add_blank": add_blank,
        "cleaned_text": cleaned_text,
        "text_cleaners": list(text_cleaners) if text_cleaners is not None else None,
        "n_speakers": n_speakers,
        "zero_shot_speakers": zero_shot_speakers,
        "frontend": frontend,
        "phonemizer_checkpoint": phonemizer_checkpoint,
        "phonemizer_lang": phonemizer_lang,
    }
    merged.update({key: value for key, value in overrides.items() if value is not None})

    return InferenceConfig(
        sampling_rate=int(merged.get("sampling_rate", 22050)),
        add_blank=bool(merged.get("add_blank", True)),
        cleaned_text=bool(merged.get("cleaned_text", True)),
        text_cleaners=_normalize_cleaners(merged.get("text_cleaners")),
        n_speakers=int(merged.get("n_speakers", 0)),
        zero_shot_speakers=bool(merged.get("zero_shot_speakers", False)),
        frontend=merged.get("frontend"),
        phonemizer_checkpoint=merged.get("phonemizer_checkpoint"),
        phonemizer_lang=merged.get("phonemizer_lang"),
    )
