import random

random.seed(42)

error_rate = 0.1
path = "./data/conllpp_train.txt"
out_path = "./data/conllpp_train_wrong_2.txt"
with open(path, "r", encoding="utf-8") as f:
    row_data = f.readlines()

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
ner_label = ["O", "B-PER", "B-ORG", "B-LOC", "B-MISC"]


def key_to_val(key, dic):
    return dic[key] if key in dic else dic["UNK"]


def get_wrong_label(correct_label):
    if correct_label[0] == "B" or correct_label == "O":
        candidates = [label for label in ner_label if label != correct_label]
    else:
        candidates = ner_label
    return random.choice(candidates)


tokens = []
labels = []
sentences = {}
data_num = 0
label_id_to_sentence = []

for line in row_data:
    if "-DOCSTART-" in line:
        continue

    elif len(line) <= 5:

        sentence = " ".join(tokens)
        if sentence not in sentences.keys():
            sentences[sentence] = {
                "labels": labels,
                "count": 1,
                "label_ids": range(len(label_id_to_sentence), len(label_id_to_sentence) + len(labels)),
            }
            label_id_to_sentence += [(sentence, i) for i in range(len(labels))]
        else:
            sentences[sentence]["count"] += 1
        tokens = []
        labels = []
        continue

    else:
        line = line.strip().split()
        tokens.append(line[0])
        labels.append(line[-1])
        data_num += 1

if len(tokens) != 0:
    sentence = " ".join(tokens)
    if sentence not in sentences.keys():
        sentences[sentence] = {
            "labels": labels,
            "count": 1,
            "label_ids": range(len(label_id_to_sentence), len(label_id_to_sentence) + len(labels)),
        }
        label_id_to_sentence += labels
    else:
        label_id_to_sentence += [(sentence, i) for i in range(len(labels))]

error_num = int(data_num * error_rate)
errors = random.sample(range(len(label_id_to_sentence)), k=error_num)

changed = 0


for i in errors:
    change_labels = sentences[label_id_to_sentence[i][0]]["labels"]
    label_idx = label_id_to_sentence[i][1]
    change_label = change_labels[label_idx]
    count = sentences[label_id_to_sentence[i][0]]["count"]
    wrong_label = get_wrong_label(change_label)

    if change_label == "O":
        if len(change_labels) > label_idx + 1 and change_labels[label_idx + 1] == wrong_label:
            change_labels[label_idx + 1] = "I" + wrong_label[1:]
            changed += count

    else:
        if wrong_label == "O":
            if len(change_labels) > label_idx + 1 and change_labels[label_idx + 1][0] == "I":
                change_labels[label_idx + 1] = "B" + change_labels[label_idx + 1][1:]
                changed += count
        else:
            j = 1
            while len(change_labels) > label_idx + j and change_labels[label_idx + j][0] == "I":
                change_labels[label_idx + j] = "I" + wrong_label[1:]
                changed += count
                j += 1
    change_labels[label_idx] = wrong_label
    changed += count

    sentences[label_id_to_sentence[i][0]]["labels"] = change_labels
    if changed >= error_num:
        break

out = []
data1 = []
data2 = []
tokens = []
labels = []
for line in row_data:
    if "-DOCSTART-" in line:
        out.append(line.strip())

    elif len(line) <= 5:
        sentence = " ".join(tokens)
        labels = sentences[sentence]["labels"]
        for token, d1, d2, label in zip(tokens, data1, data2, labels):
            out.append(f"{token} {d1} {d2} {label}")
        tokens, data1, data2 = [], [], []
        out.append(line.strip())

    else:
        line = line.strip().split()
        tokens.append(line[0])
        data1.append(line[1])
        data2.append(line[2])

out = "\n".join(out)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out)
