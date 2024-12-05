import argparse
import os
import random
import sys
import time
from logging import getLogger

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from classification.utils import get_dataset, get_inputs
from mylogger import set_logger
from roberta.bpe_dropout import RobertaTokenizerDropout
from roberta.get_sub_seq import get_subword_sequences

set_logger()
root_path = os.getcwd()
logger = getLogger(__name__)


def pred(
    tokenizer,
    sentences,
    labels,
    model,
    batch_size=16,
    device="cuda",
    length=512,
    n=500,
    k=10,
    method="k-means",
    tfidf=True,
):
    output = []

    model.eval()
    with torch.no_grad():
        for sentence, label in zip(tqdm(sentences, leave=False), labels):
            subwords = get_subword_sequences(
                tokenizer,
                sentence,
                n=n,
                out=k,
                method=method,
                tfidf=tfidf,
            )
            inputs = []
            masks = []
            true_label = []
            for i in range(k):
                test_inputs = get_inputs(
                    sentence,
                    tokenizer,
                    padding=length,
                    subwords=subwords[i][0],
                )
                inputs.append(test_inputs["input_ids"])
                masks.append(test_inputs["attention_mask"])
                true_label.append(label)
            inputs = torch.vstack(inputs).to(device)
            masks = torch.vstack(masks).to(device)
            out = []
            for batch_start in range(0, inputs.shape[0], batch_size):
                batch_end = batch_start + batch_size
                model_output = model(
                    inputs[batch_start:batch_end],
                    masks[batch_start:batch_end],
                    torch.zeros_like(masks[batch_start:batch_end]),
                ).logits
                out.append(model_output.argmax(-1).to("cpu"))
            out = torch.vstack(out).squeeze().tolist()
            if type(out) is list:
                out = [str(o) for o in out]
                out = "\t".join(out)
            else:
                out = str(out)

            out = f"{sentence}\n{label}\n{out}\n\n"
            output.append(out)

    output = "".join(output)
    return output


def main(
    test_path,
    model_path,
    output_path,
    model_name,
    device="cuda",
    length=510,
    p=0.1,
    n=500,
    k=10,
    method="k-means",
    tfidf=True,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.makedirs("./model", exist_ok=True)

    logger.info("Predict Start")
    start = time.time()

    tokenizer = RobertaTokenizerDropout.from_pretrained(model_name, p=p)
    sentences, labels, _ = get_dataset(os.path.join(root_path, test_path), sep_token=tokenizer.sep_token)

    local_model = os.path.join(root_path, model_path)
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load(local_model))
    model = model.to(device)

    out = pred(
        tokenizer,
        sentences,
        labels,
        model,
        device=device,
        length=length,
        n=n,
        k=k,
        method=method,
        tfidf=tfidf,
    )
    final_time = time.time()
    hours = (final_time - start) // 3600
    minutes = (final_time - start) // 60 - hours * 60
    seconds = (final_time - start) % 60
    logger.info(f"Pred Time: {hours}h {minutes}m {seconds}s")

    with open(os.path.join(root_path, output_path), "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/mrpc_valid.txt")
    parser.add_argument("--model_path", default="model/mrpc_model_1_baseline_fixed.pth")
    parser.add_argument("--output_path", default="outputs/predict_mrpc_train.txt")
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--length", type=int, default=510)
    parser.add_argument("--num_p", type=float, default=0)
    parser.add_argument("--num_n", type=int, default=1)
    parser.add_argument("--num_k", type=int, default=1)
    parser.add_argument("--method", default="random", choices=["random", "cos-sim", "k-means"])
    parser.add_argument("--simple_bow", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        test_path=args.test_path,
        model_path=args.model_path,
        output_path=args.output_path,
        model_name=args.model_name,
        device=args.device,
        length=args.length,
        p=args.num_p,
        n=args.num_n,
        k=args.num_k,
        method=args.method,
        tfidf=not args.simple_bow,
        seed=args.seed,
    )
