import itertools
import os
import unicodedata
from logging import getLogger

from seqeval.scheme import IOB2, Entities
from tqdm import tqdm

id2label = {0: "O", 1: "MISC", 2: "PER", 3: "ORG", 4: "LOC"}
label2id = {"LOC": 4, "MISC": 1, "ORG": 3, "PER": 2, "O": 0}
root_path = os.getcwd()
logger = getLogger(__name__)


def is_floatable(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


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


def load_documents(dataset_file):
    documents = []
    words = []
    labels = []
    weights = []
    sentence_boundaries = []
    with open(dataset_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(
                        dict(
                            words=words,
                            labels=labels,
                            sentence_boundaries=sentence_boundaries,
                            weights=weights,
                        )
                    )
                    words = []
                    labels = []
                    weights = []
                    sentence_boundaries = []
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
                    sentence_start = True
            else:
                items = line.split(" ")
                words.append(items[0])

                if is_floatable(items[-1]):
                    weight = float(items[-1])
                    labels.append(items[-2])
                else:
                    weight = 1.0
                    labels.append(items[-1])
                if sentence_start:
                    weights.append(weight)
                    sentence_start = False

    if words:
        documents.append(dict(words=words, labels=labels, sentence_boundaries=sentence_boundaries, weights=weights))

    return documents


def load_examples(documents, tokenizer, max_token_length=510, max_mention_length=30):
    examples = []

    for document in tqdm(documents):
        words = document["words"]
        subword_lengths = [len(tokenizer.tokenize(w)) for w in words]
        sentence_boundaries = document["sentence_boundaries"]

        token2subword = [0] + list(itertools.accumulate(subword_lengths))
        entities = []
        for s, e in zip(sentence_boundaries[:-1], sentence_boundaries[1:]):
            label = [boi1_to_2(document["labels"][s:e])]
            for ent in Entities(label, scheme=IOB2).entities[0]:
                ent.start += s
                ent.end += s
                entities.append(ent)

        span_to_entity_label = dict()
        for ent in entities:
            subword_start = token2subword[ent.start]
            subword_end = token2subword[ent.end]
            span_to_entity_label[(subword_start, subword_end)] = ent.tag

        for i in range(len(sentence_boundaries) - 1):
            sentence_start, sentence_end = sentence_boundaries[i : i + 2]

            text = ""

            sentence_words = words[sentence_start:sentence_end]
            sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

            word_start_char_positions = []
            word_end_char_positions = []
            for word in sentence_words:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "

            text = text.rstrip()
            entity_spans = []
            original_word_spans = []
            labels = []
            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if sum(sentence_subword_lengths[word_start : word_end + 1]) <= max_mention_length:
                        entity_spans.append(
                            (
                                word_start_char_positions[word_start],
                                word_end_char_positions[word_end],
                            )
                        )
                        original_word_spans.append((word_start, word_end + 1))
                        labels.append(span_to_entity_label.pop((sentence_start, sentence_end), "O"))
            if len(entity_spans) == 0:
                entity_spans.append(
                    (
                        word_start_char_positions[0],
                        word_end_char_positions[0],
                    )
                )
                original_word_spans.append((0, 1))
                labels.append("O")
            weights = document["weights"][i]
            examples.append(
                dict(
                    text=text,
                    words=sentence_words,
                    entity_spans=entity_spans,
                    original_word_spans=original_word_spans,
                    labels=labels,
                    weights=weights,
                )
            )

    return examples


def is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
