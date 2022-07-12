from collections import OrderedDict, Counter
from typing import List, Dict, Callable

import torchtext.vocab
from torchtext.vocab import vocab as vc


def make_class_index_mapping(labels: List[str]) -> Dict[str, int]:
    """Make a mapping that maps the classes to integral indices.

    :param labels: Label strings.
    :return: Label indices.
    """

    classes = list(set(labels))
    class_index_mapping = {
        class_: idx
        for idx, class_ in enumerate(classes)
    }

    return class_index_mapping


def make_vocab(
        texts: List[str], tokenizer: Callable,
        min_freq: int = 1, unk_token: str = "<unk>", unk_idx: int = 0,
) -> torchtext.vocab.Vocab:
    """Make a vocabulary from the specified texts.

    :param texts: Texts for making vocab.
    :param tokenizer: Tokenizer for tokenizing texts.
    :param min_freq: Min frequency of the words to be added to the vocab.
    :param unk_token: Unknown token.
    :param unk_idx: Unknown token index.
    :return: Constructed vocab.
    """

    def make_token_count_mapping() -> OrderedDict:
        """Make a mapping {token: count}"""

        all_tokens = [
            token
            for text in texts
            for token in tokenizer(text)
        ]
        counter = Counter(all_tokens)
        counter = sorted(
            counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
        counter = OrderedDict(counter)  # Not necessary for python >= 3.6

        return counter

    vocab = vc(
        ordered_dict=make_token_count_mapping(),
        min_freq=min_freq,
    )
    vocab.insert_token(
        token=unk_token,
        index=unk_idx
    )
    vocab.set_default_index(index=unk_idx)

    return vocab


def pad(token_indices: List[int], max_len: int, default_pad_val: int = 0) -> List[int]:
    """Pad the given token indices.

    :param token_indices: Indices of the tokens.
    :param max_len: Max sentence length.
    :param default_pad_val: Default padding value.
    :return: Padded indices.
    """

    if len(token_indices) < max_len:
        return token_indices + ([default_pad_val] * (max_len - len(token_indices)))
    else:
        return token_indices[:max_len]
