from typing import Iterable, List, Sequence

from symbols import BOS_ID, EOS_ID, symbols

_symbol_to_id = {symbol: index for index, symbol in enumerate(symbols)}
_id_to_symbol = {index: symbol for index, symbol in enumerate(symbols)}
_warned_unsupported_symbols = set()
_normalized_symbol_map = {
    "-": " ",
    "‐": " ",
    "‑": " ",
    "–": " ",
    "—": " ",
}


def intersperse(items: Sequence[int], filler: int) -> List[int]:
    result = [filler] * (len(items) * 2 + 1)
    result[1::2] = items
    return result


def cleaned_text_to_sequence(cleaned_text: str, strict: bool = True) -> List[int]:
    _ = strict
    missing = []
    sequence = []

    for symbol in cleaned_text:
        normalized_symbol = _normalized_symbol_map.get(symbol, symbol)
        if normalized_symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[normalized_symbol])
        else:
            missing.append(symbol)

    if missing:
        unknown_symbols = sorted(set(missing))
        fresh_symbols = [symbol for symbol in unknown_symbols if symbol not in _warned_unsupported_symbols]
        if fresh_symbols:
            _warned_unsupported_symbols.update(fresh_symbols)
            unknown = ", ".join(ascii(symbol) for symbol in fresh_symbols)
            print(f"Warning: ignoring unsupported phoneme symbols: {unknown}")

    return [BOS_ID] + sequence + [EOS_ID]


def sequence_to_text(sequence: Iterable[int]) -> str:
    return "".join(_id_to_symbol[index] for index in sequence)


class Tokenizer:
    def __init__(self, add_blank: bool = True, strict: bool = True):
        self.add_blank = add_blank
        self.strict = strict

    def encode_phonemes(self, phonemes: str) -> List[int]:
        sequence = cleaned_text_to_sequence(phonemes, strict=self.strict)
        if self.add_blank:
            sequence = intersperse(sequence, 0)
        return sequence

    def decode(self, sequence: Iterable[int]) -> str:
        return sequence_to_text(sequence)
