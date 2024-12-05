import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from roberta.utils import boi1_to_2

root_path = os.getcwd()


def main(min_weight, predict_path, train_path, output_path):
    with open(os.path.join(root_path, predict_path)) as f:
        preds = f.readlines()

    with open(os.path.join(root_path, train_path), "r", encoding="utf-8") as f:
        row_data = f.readlines()

    skip_num = 0
    golden_labels = []
    pred_labels = []
    tokens = []
    token = []
    golden_label = []
    pred_label = []
    for i, line in enumerate(row_data):
        if line.startswith("-DOCSTART-"):
            token.append("-DOCSTART-")
            golden_label.append("O")
            pred_label.append("O")
            skip_num += 2

        elif len(line) <= 2:
            tokens.append(token)
            golden_labels.append(boi1_to_2(golden_label))
            pred_labels.append(pred_label)
            token = []
            golden_label = []
            pred_label = []

        elif len(line) >= 2:
            line = line.strip().split()
            token.append(line[0])
            golden_label.append(line[-1])
            pred_label.append(preds[i - skip_num].strip().split()[1:])

    out = ""
    for token, golden_label, pred_label in zip(tokens, golden_labels, pred_labels):
        if "-DOCSTART-" in token:
            out += "-DOCSTART- -X- -X- O\n"
        else:
            pred_label = [list(x) for x in zip(*pred_label)]
            weight = max(pred_label.count(golden_label) / len(pred_label), min_weight)
            for tok, lbl in zip(token, golden_label):
                out += f"{tok} -X- -X- {lbl} {weight}\n"
        out += "\n"

    with open(os.path.join(root_path, output_path), "w") as f:
        f.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_weight", type=float, default=1 / 3)
    parser.add_argument("--predict_path", default="outputs/predict_10_k-means.txt")
    parser.add_argument("--train_path", default="data/conllpp_train.txt")
    parser.add_argument("--output_path", default="outputs/conll_fixed_10_k-means.txt")
    args = parser.parse_args()
    print(vars(args))
    main(
        args.min_weight,
        args.predict_path,
        args.train_path,
        args.output_path,
    )
