import copy
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

root_path = os.getcwd()
ner_dict = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
    "PAD": 9,
}
wnut_dict = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
    "PAD": 13,
}


class BertDataset(Dataset):
    def __init__(self, data) -> None:
        X, mask, type_ids, weights, y, predict_y = (
            data["input_ids"],
            data["attention_mask"],
            data["token_type_ids"],
            data["weights"],
            data["subword_labels"],
            data["predict_labels"],
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
        if type(predict_y) is not torch.Tensor:
            predict_y = torch.tensor(predict_y)

        self.X = X
        self.mask = mask
        self.type_ids = type_ids
        self.weights = weights
        self.y = y
        self.predict_y = predict_y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.mask[idx],
            self.type_ids[idx],
            self.weights[idx],
            self.y[idx],
            self.predict_y[idx],
        )


def get_texts_and_labels(dataset):
    tokens = dataset["tokens"]
    labels = dataset["ner_tags"]
    data = {
        "tokens": tokens,
        "labels": labels,
    }

    return data


def get_dataloader(data, batch_size, shuffle=True, drop_last=True):
    dataset = BertDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def key_to_val(key, dic):
    return dic[key] if key in dic else dic["UNK"]


def val_to_key(val, dic, pad_key="PAD"):
    keys = [k for k, v in dic.items() if v == val]
    if keys:
        return keys[0]
    return "UNK"


def boi1_to_2(labels):
    new_labels = []
    pre_tag = ""
    for label in labels:
        if label[0] == "I":
            if pre_tag in ["", "O"]:
                label = "B" + label[1:]
            elif len(pre_tag) > 4 and pre_tag[-3:] != label[-3:]:
                label = "B" + label[1:]

        new_labels.append(label)
        pre_tag = label

    return new_labels


def path_to_data(path):
    with open(path, "r", encoding="utf-8") as f:
        row_data = f.readlines()

    data = []
    doc_index = []
    tokens = []
    labels = []
    weights = []
    pre_doc_end = 0
    if "wnut" in path:
        for line in row_data:
            if len(line) <= 2:
                if len(tokens) != 0:
                    labels = [key_to_val(la, wnut_dict) for la in labels]
                    document = dict(tokens=tokens, labels=labels, doc_index=[(0, len(tokens))], weights=weights)
                    data.append(document)
                    tokens = []
                    labels = []
                    weights = []

            elif len(line) >= 2:
                line = line.strip().split()
                if line[0].startswith("http"):
                    line[0] = "<URL>"
                tokens.append(line[0])
                if line[-1] in wnut_dict.keys():
                    labels.append(line[-1])
                    weights.append(1.0)
                else:
                    weights.append(float(line[-1]))
                    labels.append(line[-2])

        if len(tokens) != 0:
            labels = [key_to_val(la, wnut_dict) for la in labels]
            document = dict(tokens=tokens, labels=labels, doc_index=[(0, len(tokens))], weights=weights)
            data.append(document)

    else:
        for line in row_data:
            if "-DOCSTART-" in line:
                if len(tokens) != 0:
                    labels = [key_to_val(la, ner_dict) for la in labels]
                    document = dict(tokens=tokens, labels=labels, doc_index=doc_index, weights=weights)
                    data.append(document)
                    tokens = []
                    labels = []
                    doc_index = []
                    weights = []
                    pre_doc_end = 0

            elif len(line) <= 5:
                if len(labels) != 0:
                    doc_start = pre_doc_end
                    doc_end = len(tokens)
                    doc_index.append((doc_start, doc_end))
                    pre_doc_end = doc_end
            else:
                line = line.strip().split()
                tokens.append(line[0])
                if line[-1] in ner_dict.keys():
                    labels.append(line[-1])
                    weights.append(1.0)
                else:
                    weights.append(float(line[-1]))
                    labels.append(line[-2])

        if len(tokens) != 0:
            labels = [key_to_val(la, ner_dict) for la in labels]
            document = dict(tokens=tokens, labels=labels, doc_index=doc_index, weights=weights)
            data.append(document)
    return data


def get_subword_label_id(label_id):
    if label_id % 2 == 0:
        return label_id
    else:
        return label_id + 1


def get_label(word_ids, label, subword_label, weights=None, wnut=False):
    label_dict = wnut_dict if wnut else ner_dict
    previous_word_idx = -100
    label_ids = []
    token_weights = []
    for word_idx in word_ids:
        if word_idx == -100:
            label_ids.append(label_dict["PAD"])
            if weights is not None:
                token_weights.append(0.0)
        elif word_idx != previous_word_idx:
            label_ids.append(label[word_idx])
            if weights is not None:
                token_weights.append(weights[word_idx])

        else:
            if subword_label == "I":
                label_ids.append(get_subword_label_id(label[word_idx]))
                if weights is not None:
                    token_weights.append(weights[word_idx])
            elif subword_label == "B":
                label_ids.append(label[word_idx])
                if weights is not None:
                    token_weights.append(weights[word_idx])
            elif subword_label == "PAD":
                label_ids.append(label_dict["PAD"])
                if weights is not None:
                    token_weights.append(0.0)
            else:
                print("subword_label must be 'I', 'B' or 'PAD'.")
                exit(1)

        previous_word_idx = word_idx
    if weights is not None:
        return label_ids, token_weights
    return label_ids


def dataset_encode(
    tokenizer,
    data,
    p=None,
    padding=512,
    return_tensor=True,
    subword_label="I",
    post_sentence_padding=False,
    add_sep_between_sentences=False,
    wnut=False,
):

    if p is None or p == 0:
        tokenizer.const_tokenize()
    else:
        tokenizer.random_tokenize()

    row_tokens = []
    row_labels = []
    input_ids = []
    attention_mask = []
    subword_labels = []
    predict_labels = []
    token_weights = []
    for document in tqdm(data):
        text = document["tokens"]
        labels = document["labels"]
        max_length = padding - 2 if padding else len(text)

        for i, j in document["doc_index"]:
            subwords, word_ids = tokenizer.tokenizeSentence(" ".join(text[i:j]))
            row_tokens.append(text[i:j])
            row_labels.append(labels[i:j])
            masked_ids = copy.deepcopy(word_ids)

            if post_sentence_padding:
                while len(subwords) < max_length and j < len(text):
                    if add_sep_between_sentences and j in [d[0] for d in document["doc_index"]]:
                        subwords.append(tokenizer.sep_token)
                        word_ids.append(-100)
                        masked_ids.append(-100)
                    ex_subwords = tokenizer.tokenize(" " + text[j])
                    subwords = subwords + ex_subwords
                    word_ids = word_ids + [max(word_ids) + 1] * len(ex_subwords)
                    masked_ids = masked_ids + [-100] * len(ex_subwords)
                    j += 1
                    if len(subwords) < max_length:
                        subwords = subwords[:max_length]
                        word_ids = word_ids[:max_length]
                        masked_ids = masked_ids[:max_length]
            subwords = (
                [tokenizer.cls_token_id]
                + [tokenizer._convert_token_to_id(w) for w in subwords]
                + [tokenizer.sep_token_id]
            )
            word_ids = [-100] + word_ids + [-100]
            masked_ids = [-100] + masked_ids + [-100]

            if len(subwords) >= padding:
                subwords = subwords[:padding]
                word_ids = word_ids[:padding]
                masked_ids = masked_ids[:padding]
                mask = [1] * padding

            else:
                attention_len = len(subwords)
                pad_len = padding - len(subwords)
                subwords += [tokenizer.pad_token_id] * pad_len
                word_ids += [-100] * pad_len
                masked_ids += [-100] * pad_len
                mask = [1] * attention_len + [0] * pad_len

            input_ids.append(subwords)
            attention_mask.append(mask)

            label = labels[i:j]
            label_ids, weights = get_label(word_ids, label, subword_label, weights=document["weights"][i:j], wnut=wnut)
            subword_labels.append(label_ids)

            masked_label = row_labels[-1]
            masked_label_ids = get_label(masked_ids, masked_label, "PAD", wnut=wnut)
            predict_labels.append(masked_label_ids)
            token_weights.append(weights)

    tokenizer.random_tokenize()

    if return_tensor:
        data = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int),
            "subword_labels": torch.tensor(subword_labels, dtype=torch.long),
            "predict_labels": torch.tensor(predict_labels, dtype=torch.long),
            "tokens": row_tokens,
            "labels": row_labels,
            "weights": torch.tensor(token_weights, dtype=torch.float32),
        }
        data["token_type_ids"] = torch.zeros_like(data["attention_mask"])
    else:
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "subword_labels": subword_labels,
            "predict_labels": predict_labels,
            "tokens": row_tokens,
            "labels": row_labels,
            "weights": token_weights,
        }
    return data


def get_inputs(
    text,
    labels,
    tokenizer,
    padding,
    subword_label,
    subwords=None,
    word_ids=None,
    wnut=False,
):
    if subwords is None or word_ids is None:
        subwords, word_ids = tokenizer.tokenizeSentence(" ".join(text))

    row_tokens = text
    row_labels = labels
    masked_ids = copy.deepcopy(word_ids)

    subwords = (
        [tokenizer.cls_token_id] + [tokenizer._convert_token_to_id(w) for w in subwords] + [tokenizer.sep_token_id]
    )
    word_ids = [-100] + word_ids + [-100]
    masked_ids = [-100] + masked_ids + [-100]

    if len(subwords) >= padding:
        subwords = subwords[:padding]
        word_ids = word_ids[:padding]
        masked_ids = masked_ids[:padding]
        mask = [1] * padding

    else:
        attention_len = len(subwords)
        pad_len = padding - len(subwords)
        subwords += [tokenizer.pad_token_id] * pad_len
        word_ids += [-100] * pad_len
        masked_ids += [-100] * pad_len
        mask = [1] * attention_len + [0] * pad_len

    label = labels
    label_ids = get_label(word_ids, label, subword_label, wnut=wnut)

    masked_label = row_labels
    masked_label_ids = get_label(masked_ids, masked_label, "PAD", wnut=wnut)

    data = {
        "input_ids": torch.tensor([subwords], dtype=torch.int),
        "attention_mask": torch.tensor([mask], dtype=torch.int),
        "subword_labels": torch.tensor([label_ids], dtype=torch.long),
        "predict_labels": torch.tensor([masked_label_ids], dtype=torch.long),
        "tokens": row_tokens,
        "labels": row_labels,
    }
    data["token_type_ids"] = torch.zeros_like(data["attention_mask"])
    return data
