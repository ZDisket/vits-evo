from dataclasses import dataclass
from typing import Iterable, Optional, cast

from cleaners import apply_cleaners, collapse_whitespace

DEFAULT_DP_CHECKPOINT = "en_us_cmudict_ipa_forward.pt"
DEFAULT_DP_LANG = "en_us"


class TextFrontend:
    def phonemize(self, text: str) -> str:
        raise NotImplementedError


@dataclass
class DeepPhonemizerFrontend(TextFrontend):
    checkpoint_path: str = DEFAULT_DP_CHECKPOINT
    lang: str = DEFAULT_DP_LANG

    def __post_init__(self):
        try:
            from dp.phonemizer import Phonemizer
        except ImportError as exc:
            raise ImportError("Install 'dp' to use the DeepPhonemizer frontend.") from exc
        self._phonemizer = Phonemizer.from_checkpoint(self.checkpoint_path)
        self._phonemizer.lang_phoneme_dict = None

    def phonemize(self, text: str) -> str:
        normalized = collapse_whitespace(text)
        if not isinstance(normalized, str):
            raise TypeError("Expected a single text string.")
        return cast(str, collapse_whitespace(self._phonemizer(normalized.lower(), lang=self.lang)))


@dataclass
class CleanersFrontend(TextFrontend):
    cleaner_names: Iterable[str]

    def phonemize(self, text: str) -> str:
        return apply_cleaners(text, self.cleaner_names)


class IdentityFrontend(TextFrontend):
    def phonemize(self, text: str) -> str:
        return cast(str, collapse_whitespace(text))


def build_frontend(
    kind: str,
    *,
    cleaner_names: Optional[Iterable[str]] = None,
    phonemizer_checkpoint: Optional[str] = None,
    phonemizer_lang: Optional[str] = None,
) -> TextFrontend:
    normalized = kind.lower().replace("-", "_")

    if normalized in {"phonemes", "prephonemized", "identity"}:
        return IdentityFrontend()
    if normalized == "deepphonemizer":
        return DeepPhonemizerFrontend(
            checkpoint_path=phonemizer_checkpoint or DEFAULT_DP_CHECKPOINT,
            lang=phonemizer_lang or DEFAULT_DP_LANG,
        )
    if normalized == "cleaners":
        if not cleaner_names:
            raise ValueError("cleaner_names are required when frontend='cleaners'.")
        return CleanersFrontend(cleaner_names=cleaner_names)

    raise ValueError(f"Unsupported frontend: {kind}")
