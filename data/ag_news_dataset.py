import csv
import torch
import torchtext.vocab
from torch.utils.data import Dataset
from typing import Dict, Callable, Tuple, List

from utils.data_utils import pad


def collate_func(
        samples: List[Tuple[str, str]],
        vocab: torchtext.vocab.Vocab, tokenizer: Callable, max_len: int,
        class_index_mapping: Dict[str, int]
) -> Dict[str, torch.tensor]:
    """Collate function for training / validation / testing.

    :param samples: A list of (text, label).
    :param vocab: Vocabulary.
    :param tokenizer: Tokenizer.
    :param max_len: Max len of the sentences.
    :param class_index_mapping: Mapping from label string to label index.
    :return: Collated batch.
    """

    texts, labels = list(zip(*samples))

    texts = torch.tensor(list(map(
        lambda text: pad(
            vocab(tokenizer(text)),
            max_len=max_len
        ),
        texts
    )))

    labels = torch.tensor(list(map(
        lambda label: class_index_mapping[label],
        labels
    )))

    return {
        "texts": texts,
        "labels": labels,
    }


class AGNewsDataset(Dataset):
    """Dataset for AG News.

    :param fp: File path of the data.
    """

    def __init__(self, fp: str):
        self.texts, self.labels = self.read(fp)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

    @classmethod
    def read(cls, fp: str) -> Tuple[List[str], List[str]]:
        """Obtain texts and labels from the data file.

        :param fp: File path of the data.
        :return: texts and labels.
        """

        texts = []
        labels = []
        with open(fp, encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                label, title, body = line

                # Concatenate the title and the body as text.
                texts.append(f"{title} {body}")
                labels.append(label)

                # TODO: Some operations in collate_func can be placed here.

        return texts, labels
