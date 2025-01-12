import argparse
import os
import random
import sys
import time
from logging import getLogger

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from classification.trainer import trainer
from classification.utils import dataset_encode, get_dataloader, get_dataset
from mylogger import set_logger
from roberta.bpe_dropout import RobertaTokenizerDropout

set_logger()
root_path = os.getcwd()
logger = getLogger(__name__)


def main(
    train_path,
    output_path,
    batch_size=4,
    accum_iter=8,
    lr=1e-5,
    num_epoch=5,
    length=510,
    model_name="roberta-large",
    weight_decay=0.01,
    use_scheduler=True,
    warmup_late=0.1,
    device="cuda",
    seed=42,
    mid_output_epoch=2,
    valid_path=None,
    test_path=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.makedirs("./model", exist_ok=True)

    logger.info("Train Start")
    start = time.time()

    tokeninzer = RobertaTokenizerDropout.from_pretrained(model_name, p=0)
    sentences, labels, weights = get_dataset(train_path, sep_token=tokeninzer.sep_token)
    train_data = dataset_encode(sentences, labels, weights, tokeninzer, padding=length)
    train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)

    if valid_path is not None:
        sentences, labels, weights = get_dataset(valid_path, sep_token=tokeninzer.sep_token)
        valid_data = dataset_encode(sentences, labels, weights, tokeninzer, padding=length)
        valid_loader = get_dataloader(valid_data, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        valid_loader = None

    if test_path is not None:
        sentences, labels, weights = get_dataset(test_path, sep_token=tokeninzer.sep_token)
        test_data = dataset_encode(sentences, labels, weights, tokeninzer, padding=length)
        test_loader = get_dataloader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        test_loader = None

    num_training_steps = int(len(train_loader) / accum_iter) * num_epoch
    num_warmup_steps = int(num_training_steps * warmup_late)

    net = trainer(
        model_name=model_name,
        lr=lr,
        batch_size=batch_size,
        length=length,
        accum_iter=accum_iter,
        weight_decay=weight_decay,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        use_scheduler=use_scheduler,
        device=device,
        output_path=output_path,
        mid_output_epoch=mid_output_epoch,
    )
    net.train(train_loader, num_epoch, valid_loader=valid_loader, test_loader=test_loader)

    train_time = time.time()
    hours = (train_time - start) // 3600
    minutes = (train_time - start) // 60 - hours * 60
    seconds = (train_time - start) % 60
    logger.info(f"Train Time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/mrpc_train.txt")
    parser.add_argument("--valid_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--output_path", default="model/mrpc_model.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_iter", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--length", type=int, default=510)
    parser.add_argument("--model_name", default="roberta-large")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--without_scheduler", action="store_true")
    parser.add_argument("--warmup_late", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mid_output_epoch", type=int, default=2)

    args = parser.parse_args()
    print(vars(args))
    main(
        args.train_path,
        args.output_path,
        batch_size=args.batch_size,
        accum_iter=args.accum_iter,
        lr=args.lr,
        num_epoch=args.num_epoch,
        length=args.length,
        model_name=args.model_name,
        weight_decay=args.weight_decay,
        use_scheduler=not args.without_scheduler,
        warmup_late=args.warmup_late,
        device=args.device,
        seed=args.seed,
        mid_output_epoch=args.mid_output_epoch,
        valid_path=args.valid_path,
        test_path=args.test_path,
    )
