import argparse
import os
import random
import sys
import time
from logging import getLogger

import numpy as np
import torch
from tqdm import tqdm
from transformers import LukeForEntitySpanClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from luke_predict.bpe_dropout import LukeTokenizerDropout
from luke_predict.get_sub_seq import get_subword_sequences
from luke_predict.utils import load_documents, load_examples
from mylogger import set_logger

set_logger()
root_path = os.getcwd()
logger = getLogger(__name__)
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
paddings = {
    "input_ids": 1,
    "attention_mask": 0,
}


def get_padding(input, key, device):
    input = [i.get(key) for i in input]
    max_length = max([len(i) for i in input])
    padding_input = []
    for i in input:
        if len(i) != max_length:
            i += [paddings[key]] * (max_length - len(i))
        padding_input.append(i)
    padding_input = torch.tensor(padding_input, dtype=torch.long).to(device)
    return padding_input


def get_tensor(inputs, device):
    keys = inputs[0].keys()
    inputs = {key: get_padding(inputs, key, device) for key in keys}
    return keys, inputs


def pred(
    model,
    tokenizer,
    test_examples,
    batch_size=2,
    n=500,
    k=10,
    method="random",
    tfidf=True,
    device="cuda",
):
    model.eval()
    output = []
    for example in tqdm(test_examples):
        inputs = get_subword_sequences(
            tokenizer=tokenizer,
            texts=example["text"],
            entity_spans=example["entity_spans"],
            n=n,
            out=k,
            method=method,
            tfidf=tfidf,
        )
        keys, inputs = get_tensor(inputs, device)
        all_logits = []
        for i in range(0, k, batch_size):
            batch_inputs = {}
            for key in keys:
                batch_inputs[key] = inputs[key][i : i + batch_size]
            with torch.no_grad():
                outputs = model(**batch_inputs)
            all_logits.extend(outputs.logits.tolist())

        final_predictions = []
        for logits in all_logits:
            max_logits = np.max(logits, axis=1)
            max_indices = np.argmax(logits, axis=1)
            original_spans = example["original_word_spans"]
            predictions = []
            for logit, index, span in zip(max_logits, max_indices, original_spans):
                if index != 0:
                    predictions.append((logit, span, model.config.id2label[index]))

            predicted_sequence = ["O"] * len(example["words"])
            for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
                if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                    predicted_sequence[span[0]] = "B-" + label
                    if span[1] - span[0] > 1:
                        predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

            final_predictions.append(predicted_sequence)
        preds = []
        for pred in final_predictions:
            if preds == []:
                preds = pred
            else:
                for j, p in enumerate(pred):
                    preds[j] += f"\t{p}"
        out = "\n".join([f"{token}\t{pred}" for token, pred in zip(example["words"], preds)])
        output.append(out)

    output = "\n\n".join(output)
    return output


def main(
    test_path,
    model_path,
    output_path,
    model_name,
    device="cuda",
    p=0.1,
    n=500,
    k=10,
    batch_size=2,
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

    tokenizer = LukeTokenizerDropout.from_pretrained(model_name, p=p)
    test_documents = load_documents(os.path.join(root_path, test_path))[:10]
    test_examples = load_examples(test_documents, tokenizer, max_mention_length=16)

    local_model = os.path.join(root_path, model_path)
    if os.path.exists(local_model):
        model = LukeForEntitySpanClassification.from_pretrained(local_model, local_files_only=True)
    else:
        model = LukeForEntitySpanClassification.from_pretrained(model_name)
    model = model.to(device)

    out = pred(
        model,
        tokenizer,
        test_examples,
        batch_size=batch_size,
        n=n,
        k=k,
        method=method,
        tfidf=tfidf,
        device=device,
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
    parser.add_argument("--test_path", default="data/eng.train")
    parser.add_argument("--model_path", default="model/luke_large.pth")
    parser.add_argument("--output_path", default="outputs/predict_luke.txt")
    parser.add_argument("--model_name", default="studio-ousia/luke-large-finetuned-conll-2003")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_p", type=float, default=0.1)
    parser.add_argument("--num_n", type=int, default=500)
    parser.add_argument("--num_k", type=int, default=10)
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
        p=args.num_p,
        n=args.num_n,
        k=args.num_k,
        method=args.method,
        tfidf=not args.simple_bow,
        seed=args.seed,
    )
