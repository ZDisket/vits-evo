import re
from typing import Callable, Iterable, List, Sequence, Union

TextValue = Union[str, Sequence[str]]

_whitespace_re = re.compile(r"\s+")

_abbreviations = [(re.compile(r"\b%s\." % item[0], re.IGNORECASE), item[1]) for item in [
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
]]

_german_abbreviations = [(re.compile(r"\b%s\." % item[0], re.IGNORECASE), item[1]) for item in [
    ("dr", "doktor"),
    ("hr", "herr"),
    ("fr", "frau"),
    ("msc", "master of science"),
    ("bsc", "bachelor of science"),
    ("u.a", "unter anderem"),
    ("usw", "und so weiter"),
    ("z.b", "zum beispiel"),
    ("ca", "circa"),
]]

_spanish_abbreviations = [(re.compile(r"\b%s\." % item[0], re.IGNORECASE), item[1]) for item in [
    ("sr", "señor"),
    ("sra", "señora"),
    ("srta", "señorita"),
    ("dr", "doctor"),
    ("dra", "doctora"),
    ("ud", "usted"),
    ("uds", "ustedes"),
    ("etc", "etcétera"),
    ("aprox", "aproximadamente"),
    ("dept", "departamento"),
    ("ing", "ingeniero"),
    ("lic", "licenciado"),
    ("prof", "profesor"),
    ("gral", "general"),
    ("col", "coronel"),
    ("av", "avenida"),
]]


def _map_text(text: TextValue, fn: Callable[[str], str]) -> TextValue:
    if isinstance(text, str):
        return fn(text)
    return [fn(item) for item in text]


def _require_unidecode():
    try:
        from unidecode import unidecode
    except ImportError as exc:
        raise ImportError("Install 'Unidecode' to use cleaner-based raw-text inference.") from exc
    return unidecode


def _require_phonemize():
    try:
        from phonemizer import phonemize
    except ImportError as exc:
        raise ImportError(
            "Install 'phonemizer' and make sure espeak-ng is available to use cleaner-based raw-text inference."
        ) from exc
    return phonemize


def collapse_whitespace(text: TextValue) -> TextValue:
    return _map_text(text, lambda value: re.sub(_whitespace_re, " ", value).strip())


def lowercase(text: TextValue) -> TextValue:
    return _map_text(text, str.lower)


def convert_to_ascii(text: TextValue) -> TextValue:
    unidecode = _require_unidecode()
    return _map_text(text, unidecode)


def _expand_patterns(text: TextValue, patterns) -> TextValue:
    def apply(value: str) -> str:
        for regex, replacement in patterns:
            value = re.sub(regex, replacement, value)
        return value

    return _map_text(text, apply)


def expand_abbreviations(text: TextValue) -> TextValue:
    return _expand_patterns(text, _abbreviations)


def expand_german_abbreviations(text: TextValue) -> TextValue:
    return _expand_patterns(text, _german_abbreviations)


def expand_spanish_abbreviations(text: TextValue) -> TextValue:
    return _expand_patterns(text, _spanish_abbreviations)


def basic_cleaners(text: TextValue) -> TextValue:
    return collapse_whitespace(lowercase(text))


def transliteration_cleaners(text: TextValue) -> TextValue:
    return collapse_whitespace(lowercase(convert_to_ascii(text)))


def _run_espeak(
    text: TextValue,
    *,
    language: str,
    preserve_punctuation: bool = False,
    with_stress: bool = False,
    language_switch: str = None,
) -> TextValue:
    phonemize = _require_phonemize()
    kwargs = {
        "language": language,
        "backend": "espeak",
        "strip": True,
        "preserve_punctuation": preserve_punctuation,
        "with_stress": with_stress,
    }
    if language_switch is not None:
        kwargs["language_switch"] = language_switch
    return collapse_whitespace(phonemize(text, **kwargs))


def phonemize_gruut(text: TextValue, lang: str = "en-us") -> TextValue:
    try:
        from gruut import sentences
    except ImportError as exc:
        raise ImportError("Install 'gruut' to use gruut cleaner frontends.") from exc

    punctuation = ",.!;?:|‖"

    def phonemize_one(value: str) -> str:
        output = ""
        for sentence in sentences(value, lang=lang):
            for word in sentence:
                output += " "
                if word.text in punctuation:
                    output += word.text
                    continue
                if word.phonemes:
                    output += "".join(word.phonemes)
                else:
                    output += word.text
        return output.strip()

    return collapse_whitespace(_map_text(text, phonemize_one))


def english_cleaners(text: TextValue) -> TextValue:
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    return _run_espeak(text, language="en-us")


def english_cleaners2(text: TextValue) -> TextValue:
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    return _run_espeak(text, language="en-us", preserve_punctuation=True, with_stress=True)


def english_cleaners2_gruut(text: TextValue) -> TextValue:
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    return phonemize_gruut(text, lang="en-us")


def german_cleaners(text: TextValue) -> TextValue:
    text = lowercase(text)
    text = expand_german_abbreviations(text)
    return _run_espeak(
        text,
        language="de",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )


def german_cleaners_gruut(text: TextValue) -> TextValue:
    text = lowercase(text)
    text = expand_german_abbreviations(text)
    return phonemize_gruut(text, lang="de")


def chinese_cleaners(text: TextValue) -> TextValue:
    return _run_espeak(text, language="cmn", preserve_punctuation=True)


def cantonese_cleaners(text: TextValue) -> TextValue:
    return _run_espeak(text, language="yue", preserve_punctuation=True)


def spanish_cleaners(text: TextValue) -> TextValue:
    text = lowercase(text)
    text = expand_spanish_abbreviations(text)
    return _run_espeak(text, language="es", preserve_punctuation=True, with_stress=True)


_CLEANERS = {
    "basic_cleaners": basic_cleaners,
    "transliteration_cleaners": transliteration_cleaners,
    "english_cleaners": english_cleaners,
    "english_cleaners2": english_cleaners2,
    "english_cleaners2_gruut": english_cleaners2_gruut,
    "german_cleaners": german_cleaners,
    "german_cleaners_gruut": german_cleaners_gruut,
    "chinese_cleaners": chinese_cleaners,
    "cantonese_cleaners": cantonese_cleaners,
    "spanish_cleaners": spanish_cleaners,
}


def apply_cleaners(text: str, cleaner_names: Iterable[str]) -> str:
    cleaned = text
    for name in cleaner_names:
        try:
            cleaner = _CLEANERS[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported cleaner for minimal inference: {name}") from exc
        cleaned = cleaner(cleaned)
    if not isinstance(cleaned, str):
        raise TypeError("Cleaner pipeline returned a non-string value.")
    return cleaned
