import argparse
import os

root_path = os.getcwd()


def main(min_weight, predict_path, output_path):
    with open(os.path.join(root_path, predict_path), encoding="utf-8") as f:
        data = f.read()
    data = data.split("\n\n")
    out = ""
    for d in data:
        if len(d) < 3:
            continue
        d = d.split("\n")
        label = d[-2].strip()
        pred = d[-1].strip().split("\t")
        weight = max(sum([p == label for p in pred]) / len(pred), min_weight)
        out += "\n".join(d[:-1]) + f"\n{weight}\n\n"

    with open(os.path.join(root_path, output_path), "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_weight", type=float, default=1 / 3)  # 1 / 3)
    parser.add_argument("--predict_path", default="outputs/predict_mrpc_10_random.txt")
    parser.add_argument("--output_path", default="data/mrpc_10_random.txt")
    args = parser.parse_args()
    print(vars(args))
    main(
        args.min_weight,
        args.predict_path,
        args.output_path,
    )
