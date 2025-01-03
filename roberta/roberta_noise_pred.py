import argparse
import os
import random
import sys
import time
from logging import getLogger

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mylogger import set_logger
from roberta.bpe_dropout import RobertaTokenizerDropout
from roberta.get_sub_seq import get_subword_sequences
from roberta.utils import get_inputs, ner_dict, path_to_data, val_to_key
from roberta.white_noise_model import RoBERTaWithNoise

set_logger()
root_path = os.getcwd()
logger = getLogger(__name__)


def pred(
    tokenizer,
    test_dataset,
    model,
    batch_size=16,
    device="cuda",
    length=512,
    k=10,
    alpha=0.1,
):
    output = []
    label_dict = ner_dict

    model.eval()
    with torch.no_grad():
        for test_data in tqdm(test_dataset, leave=False):
            text = test_data["tokens"]
            labels = test_data["labels"]
            for sentence_start, sentence_end in test_data["doc_index"]:
                subwords = get_subword_sequences(
                    tokenizer,
                    " ".join(text[sentence_start:sentence_end]),
                    n=k,
                    out=k,
                    method="random",
                    tfidf=False,
                )
                preds = []
                inputs = []
                masks = []
                true_label = []
                for i in range(k):
                    test_inputs = get_inputs(
                        text[sentence_start:sentence_end],
                        labels[sentence_start:sentence_end],
                        tokenizer,
                        padding=length,
                        subword_label="PAD",
                        subwords=subwords[i][0],
                        word_ids=subwords[i][1],
                        wnut=False,
                    )
                    inputs.append(test_inputs["input_ids"])
                    masks.append(test_inputs["attention_mask"])
                    true_label.append(test_inputs["predict_labels"][0])
                inputs = torch.vstack(inputs).to(device)
                masks = torch.vstack(masks).to(device)
                out = []
                for batch_start in range(0, inputs.shape[0], batch_size):
                    batch_end = batch_start + batch_size
                    model_output = model(
                        inputs[batch_start:batch_end], masks[batch_start:batch_end], alpha=alpha
                    ).logits
                    out.append(model_output.argmax(-1).to("cpu"))
                out = torch.vstack(out).tolist()
                for pred, label in zip(out, true_label):

                    pred = [val_to_key(prd, label_dict) for (prd, lbl) in zip(pred, label) if lbl != label_dict["PAD"]]
                    pred = [c if c != "PAD" else "O" for c in pred]
                    if preds == []:
                        preds = pred
                    else:
                        for j, p in enumerate(pred):
                            preds[j] += f"\t{p}"

                out = "\n".join([f"{token}\t{pred}" for token, pred in zip(text[sentence_start:sentence_end], preds)])
                output.append(out)

    output = "\n\n".join(output)
    return output


def main(
    test_path,
    model_path,
    output_path,
    model_name,
    device="cuda",
    length=510,
    alpha=0.1,
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
    test_dataset = path_to_data(os.path.join(root_path, test_path))
    start = time.time()

    label_dict = ner_dict
    local_model = os.path.join(root_path, model_path)
    tokenizer = RobertaTokenizerDropout.from_pretrained(model_name, p=0)
    config = AutoConfig.from_pretrained(model_name, num_labels=len(label_dict))
    model = RoBERTaWithNoise.from_config(config)
    model.load_state_dict(torch.load(local_model))
    model = model.to(device)

    out = pred(
        tokenizer,
        test_dataset,
        model,
        device=device,
        length=length,
        k=k,
        alpha=alpha,
    )
    final_time = time.time()
    hours = (final_time - start) // 3600
    minutes = (final_time - start) // 60 - hours * 60
    seconds = (final_time - start) % 60
    logger.info(f"Pred Time: {hours}h {minutes}m {seconds}s")

    with open(os.path.join(root_path, output_path), "w") as f:
        f.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/conllpp_train.txt")
    parser.add_argument("--model_path", default="model/model.pth")
    parser.add_argument("--output_path", default="outputs/predict.txt")
    parser.add_argument("--model_name", default="roberta-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--length", type=int, default=510)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--num_n", type=int, default=500)
    parser.add_argument("--num_k", type=int, default=2)
    parser.add_argument("--method", default="random", choices=["random", "cos-sim", "k-means"])
    parser.add_argument("--simple_bow", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    alpha = args.alpha / 10
    main(
        test_path=args.test_path,
        model_path=args.model_path,
        output_path=args.output_path,
        model_name=args.model_name,
        device=args.device,
        length=args.length,
        alpha=alpha,
        n=args.num_n,
        k=args.num_k,
        method=args.method,
        tfidf=not args.simple_bow,
        seed=args.seed,
    )
