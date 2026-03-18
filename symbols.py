"""Minimal symbol set for ONNX inference."""

_pad = "_"
_punctuation = ';:,.!?¡¿—…\"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_ipa_vowels = (
    "i"
    "y"
    "ɨ"
    "ʉ"
    "ɯ"
    "u"
    "ɪ"
    "ʏ"
    "ʊ"
    "e"
    "ø"
    "ɘ"
    "ɵ"
    "ɤ"
    "o"
    "ə"
    "ɛ"
    "œ"
    "ɜ"
    "ɞ"
    "ʌ"
    "ɔ"
    "æ"
    "ɐ"
    "a"
    "ɶ"
    "ɑ"
    "ɒ"
    "ɚ"
    "ɝ"
    "ᵻ"
)

_ipa_consonants = (
    "p" "b" "t" "d" "ʈ" "ɖ" "c" "ɟ" "k" "ɡ" "q" "ɢ" "ʔ" "ʡ"
    "m" "ɱ" "n" "ɳ" "ɲ" "ŋ" "ɴ"
    "ʙ" "r" "ʀ"
    "ⱱ" "ɾ" "ɽ"
    "ɸ" "β" "f" "v" "θ" "ð" "s" "z" "ʃ" "ʒ" "ʂ" "ʐ" "ç" "ʝ"
    "x" "ɣ" "χ" "ʁ" "ħ" "ʕ" "h" "ɦ"
    "ɬ" "ɮ"
    "ʋ" "ɹ" "ɻ" "j" "ɰ" "w" "ʍ"
    "l" "ɭ" "ʎ" "ʟ"
    "ʤ" "ʧ" "ʦ" "ʣ" "ʨ" "ʥ"
    "ɓ" "ɗ" "ʄ" "ɠ" "ʛ"
    "ʘ" "ǀ" "ǁ" "ǂ" "ǃ"
    "ɕ" "ʑ" "ɧ" "ʜ" "ʢ" "ɺ" "ɥ"
    "ɫ"
)

_ipa_suprasegmentals = (
    "ˈ"
    "ˌ"
    "ː"
    "ˑ"
    "ʼ"
    "ˡ"
    "|"
    "‖"
    "‿"
    "↗"
    "↘"
    "↑"
    "↓"
    "→"
)

_ipa_diacritics = (
    "ʰ"
    "ʱ"
    "ʲ"
    "ʷ"
    "ˠ"
    "ˤ"
    "˞"
    "ʴ"
    "̃"
    "̩"
    "̯"
    "̪"
    "̰"
    "̤"
    "̥"
    "̬"
    "̚"
    "̝"
    "̞"
    "̻"
    "̼"
    "ⁿ"
    "ˀ"
    "'"
)

_ipa_tones = (
    "˥"
    "˦"
    "˧"
    "˨"
    "˩"
)

_letters_ipa = (
    _ipa_vowels
    + _ipa_consonants
    + _ipa_suprasegmentals
    + _ipa_diacritics
    + _ipa_tones
)

_bos = "<BOS>"
_eos = "<EOS>"

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [_bos, _eos]

_seen = set()
_deduped = []
for _symbol in symbols:
    if _symbol not in _seen:
        _seen.add(_symbol)
        _deduped.append(_symbol)
symbols = _deduped

SPACE_ID = symbols.index(" ")
BOS_ID = symbols.index(_bos)
EOS_ID = symbols.index(_eos)
