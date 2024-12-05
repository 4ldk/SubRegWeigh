import argparse

from sklearn.metrics import f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", default="outputs/eval_valid_sst2_1_baseline.txt")
    args = parser.parse_args()
    with open(args.eval_path, encoding="utf-8") as f:
        data = f.read()

    data = data.split("\n\n")
    correct = []
    pred = []
    true = []
    for d in data:
        if len(d) < 3:
            continue
        d = d.split("\n")
        correct.append(d[-2] == d[-1])
        true.append(int(d[-2]))
        pred.append(int(d[-1]))

    print(len(correct))
    acc = sum(correct) / len(correct)
    f1 = f1_score(true, pred)
    print(f"acc:{acc:.4f} f1:{f1:.4f}")
