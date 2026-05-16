import json
import re
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
from tokenizers import Tokenizer
from datasets import load_dataset


def basic_clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


# Fallback-подготовка WikiText, если файл из ЛР1 не найден.
def prepare_wikitext_jsonl(output_path: str, max_texts: int = 20000):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    texts = []
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            text = basic_clean_text(item["text"])
            if len(text.split()) >= 10:
                texts.append(text)

            if len(texts) >= max_texts:
                break

        if len(texts) >= max_texts:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def read_texts_from_jsonl(path: str, max_texts=None) -> List[str]:
    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            text = obj.get("text", "").strip()

            if text:
                texts.append(text)

            if max_texts is not None and len(texts) >= max_texts:
                break

    return texts


def encode_texts(
    texts: List[str],
    tokenizer: Tokenizer,
    append_eos: bool = True,
) -> List[List[int]]:
    eos_id = tokenizer.token_to_id("<EOS>")

    sequences = []

    for text in texts:
        ids = tokenizer.encode(text).ids

        if append_eos and eos_id is not None:
            ids = ids + [eos_id]

        if len(ids) > 1:
            sequences.append(ids)

    return sequences


def pack_sequences(
    sequences: List[List[int]],
    block_size: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    packed_input_ids = []
    packed_masks = []

    current_ids = []
    current_mask = []
    current_object_id = 1

    def flush():
        nonlocal current_ids, current_mask, current_object_id

        if len(current_ids) == 0:
            return

        pad_len = block_size - len(current_ids)

        current_ids = current_ids + [pad_id] * pad_len
        current_mask = current_mask + [0] * pad_len

        packed_input_ids.append(current_ids)
        packed_masks.append(current_mask)

        current_ids = []
        current_mask = []
        current_object_id = 1

    for seq in sequences:
        chunks = [
            seq[i:i + block_size]
            for i in range(0, len(seq), block_size)
        ]

        for chunk in chunks:
            if len(chunk) == 0:
                continue

            if len(current_ids) + len(chunk) > block_size:
                flush()

            current_ids.extend(chunk)
            current_mask.extend([current_object_id] * len(chunk))
            current_object_id += 1

            if len(current_ids) == block_size:
                flush()

    flush()

    return (
        torch.tensor(packed_input_ids, dtype=torch.long),
        torch.tensor(packed_masks, dtype=torch.long),
    )


class PackedLanguageModelingDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        packed_mask: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.packed_mask = packed_mask

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "packed_mask": self.packed_mask[idx],
        }


class PackedTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_path: str,
        text_file: str,
        block_size: int = 256,
        batch_size: int = 16,
        val_ratio: float = 0.1,
        num_workers: int = 0,
        max_texts: int = None,
        append_eos: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.tokenizer_path = tokenizer_path
        self.text_file = text_file
        self.block_size = block_size
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.max_texts = max_texts
        self.append_eos = append_eos
        self.seed = seed

        self.tokenizer = None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        if not Path(self.text_file).exists():
            prepare_wikitext_jsonl(
                output_path=self.text_file,
                max_texts=self.max_texts or 20000,
            )

    def setup(self, stage=None):
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        pad_id = self.tokenizer.token_to_id("<PAD>")
        if pad_id is None:
            raise ValueError("В токенизаторе должен быть специальный токен <PAD>.")

        texts = read_texts_from_jsonl(
            self.text_file,
            max_texts=self.max_texts,
        )

        sequences = encode_texts(
            texts=texts,
            tokenizer=self.tokenizer,
            append_eos=self.append_eos,
        )

        input_ids, packed_mask = pack_sequences(
            sequences=sequences,
            block_size=self.block_size,
            pad_id=pad_id,
        )

        self.dataset = PackedLanguageModelingDataset(
            input_ids=input_ids,
            packed_mask=packed_mask,
        )

        val_size = int(len(self.dataset) * self.val_ratio)
        train_size = len(self.dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )