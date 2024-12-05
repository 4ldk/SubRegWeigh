import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_dataset(data_path, sep_token="[seq]"):
    with open(data_path, encoding="utf-8") as f:
        data = f.read()
    data = data.split("\n\n")
    sentences = []
    labels = []
    weights = []
    for d in data:
        if len(d) < 3:
            continue
        d = d.split("\n")
        if len(d) == 3:
            sentences.append(d[0][:-1])
        else:
            sentences.append(f"{d[0][:-1]}{sep_token}{sep_token}{d[1][:-1]}")
        labels.append(int(d[-2]))
        weights.append(float(d[-1]))

    return sentences, labels, weights


def dataset_encode(sentences, labels, weights, tokenizer, padding=512):
    input_ids = []
    attention_mask = []
    for sentence in tqdm(sentences):
        subwords, _ = tokenizer.tokenizeSentence(sentence)
        subwords = (
            [tokenizer.cls_token_id] + [tokenizer._convert_token_to_id(w) for w in subwords] + [tokenizer.sep_token_id]
        )

        if len(subwords) >= padding:
            subwords = subwords[:padding]
            mask = [1] * padding

        else:
            attention_len = len(subwords)
            pad_len = padding - len(subwords)
            subwords += [tokenizer.pad_token_id] * pad_len
            mask = [1] * attention_len + [0] * pad_len

        input_ids.append(subwords)
        attention_mask.append(mask)
    tokenizer.random_tokenize()

    data = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int),
        "labels": torch.tensor(labels, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
    }
    data["token_type_ids"] = torch.zeros_like(data["attention_mask"])

    return data


class BertDataset(Dataset):
    def __init__(self, data) -> None:
        X, mask, type_ids, weights, y = (
            data["input_ids"],
            data["attention_mask"],
            data["token_type_ids"],
            data["weights"],
            data["labels"],
        )
        super().__init__()
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if type(mask) is not torch.Tensor:
            mask = torch.tensor(mask)
        if type(type_ids) is not torch.Tensor:
            type_ids = torch.tensor(type_ids)
        if type(weights) is not torch.Tensor:
            weights = torch.tensor(weights)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y)

        self.X = X
        self.mask = mask
        self.type_ids = type_ids
        self.weights = weights
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.mask[idx],
            self.type_ids[idx],
            self.weights[idx],
            self.y[idx],
        )


def get_dataloader(data, batch_size, shuffle=True, drop_last=True):
    dataset = BertDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def get_inputs(
    sentence,
    tokenizer,
    padding,
    subwords=None,
):
    if subwords is None:
        subwords, _ = tokenizer.tokenizeSentence(sentence)

    subwords = (
        [tokenizer.cls_token_id] + [tokenizer._convert_token_to_id(w) for w in subwords] + [tokenizer.sep_token_id]
    )

    if len(subwords) >= padding:
        subwords = subwords[:padding]
        mask = [1] * padding

    else:
        attention_len = len(subwords)
        pad_len = padding - len(subwords)
        subwords += [tokenizer.pad_token_id] * pad_len
        mask = [1] * attention_len + [0] * pad_len

    data = {
        "input_ids": torch.tensor([subwords], dtype=torch.int),
        "attention_mask": torch.tensor([mask], dtype=torch.int),
    }
    data["token_type_ids"] = torch.zeros_like(data["attention_mask"])
    return data
