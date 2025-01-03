import copy
import os


def boi2_to_1(labels):
    new_labels = []
    for label in labels:
        # ner = [O,B-LOC,I-LOC,I-LOC,B-LOC,I-LOC,B-MISC,B-MISC]
        pre_tag = -1
        new_ner = []
        for la in label:
            if la in [1, 3, 5, 7] and pre_tag not in [la + 1, la]:
                pre_tag = copy.deepcopy(la)
                la += 1
            else:
                pre_tag = copy.deepcopy(la)
            new_ner.append(copy.deepcopy(la))
        # new_ner = [O,I-LOC,I-LOC,I-LOC,B-LOC,I-LOC,I-MISC,B-MISC]
        new_labels.append(new_ner)

    return new_labels


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


if __name__ == "__main__":
    labels = ["I-ORG", "I-ORG", "O", "I-ORG", "I-MISC", "I-LOC"]
    print(boi1_to_2(labels))

with open(os.path.join(os.getcwd(), "outputs/test_weights/train.bio")) as f:
    weights = f.readlines()
    weight_data = {}
    text = []
    weight = 1.0
    for w in weights:
        if len(w) < 2:
            text = " ".join(text)
            if text in weight_data.keys() and weight_data[text] != weight:
                print(text)
                print(weight, weight_data[text])
            weight_data[text] = weight
            text = []
        else:
            w = w.split("\t")
            text.append(w[0])
            weight = w[2]

with open(os.path.join(os.getcwd(), "data/conll_test_orgi.txt")) as f:
    lines = f.readlines()

out = []
data = []
doc_index = []
tokens = []
labels = []
replaced_labels = []
pre_doc_end = 0
for line in lines:

    if "-DOCSTART-" in line:
        if len(tokens) != 0:
            document = dict(tokens=tokens, labels=replaced_labels, doc_index=doc_index)
            data.append(document)
            tokens = []
            replaced_labels = []
            labels = []
            doc_index = []
            pre_doc_end = 0

    elif len(line) <= 5:
        if len(labels) != 0:
            doc_start = pre_doc_end
            doc_end = len(tokens)
            doc_index.append((doc_start, doc_end))

            pre_doc_end = doc_end
            replaced_labels += boi1_to_2(labels)
            labels = []
    else:
        line = line.strip().split()

        tokens.append(line[0])
        labels.append(line[-1])
if len(tokens) != 0:
    document = dict(tokens=tokens, labels=replaced_labels, doc_index=doc_index)
    data.append(document)
out = []
for document in data:
    out.append("-DOCSTART- -X- O O\n\n")
    for index in document["doc_index"]:
        tokens = document["tokens"][index[0] : index[1]]
        labels = document["labels"][index[0] : index[1]]
        text = " ".join(tokens)
        weight = weight_data[text] if text in weight_data.keys() else 1.0
        for token, label in zip(tokens, labels):
            line = f"{token} -X- O {label} {weight}"
            out.append(line)
        out.append("\n")

with open(os.path.join(os.getcwd(), "outputs/conllt_fixed_10_weighed_2.txt"), "w") as f:
    f.write("".join(out))
