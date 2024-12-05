import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)


class trainer:
    def __init__(
        self,
        model_name="bert-base-cased",
        lr=1e-5,
        batch_size=16,
        length=512,
        accum_iter=2,
        weight_decay=0.01,
        num_warmup_steps=None,
        num_training_steps=None,
        use_scheduler=False,
        device="cuda",
        output_path="model/model.pt",
        mid_output_epoch=2,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)

        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        if use_scheduler:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)

        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.length = length
        self.use_scheduler = use_scheduler
        self.accum_iter = accum_iter
        self.output_path = output_path
        self.mid_output_epoch = mid_output_epoch

    def forward(self, input, mask, type_ids=None, weight=None, label=None):
        logits = self.model(input, mask, type_ids).logits
        pred = logits.argmax(-1).squeeze(-1)

        if label is not None:
            loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), label.reshape(-1)) * weight.reshape(-1)
            loss = loss.mean()
            return logits, pred.to("cpu"), loss

        return logits, pred.to("cpu")

    def step(self, batch, batch_idx, batch_num, train=True):

        input, mask, type_ids, weights, label = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
            batch[3].to(self.device),
            batch[4].to(self.device),
        )

        if train:
            logits, pred, loss = self.forward(input, mask, type_ids, weights, label)

            (loss / self.accum_iter).backward()
            if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == batch_num):
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                if self.use_scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        else:
            with torch.no_grad():
                logits, pred, loss = self.forward(input, mask, type_ids, weights, label)

        return logits, pred, loss, label.to("cpu")

    def epoch_loop(self, epoch, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        bar = tqdm(loader, leave=False, desc=f"Epoch{epoch} ")
        batch_loss, preds, labels = [], [], []
        final_loss, acc, f1 = 0, 0, 0

        for batch_idx, batch in enumerate(bar):
            logits, pred, loss, label = self.step(batch, batch_idx, batch_num=len(loader), train=train)
            bar.set_postfix(loss=loss.to("cpu").item())
            if train:
                batch_loss.append(loss.to("cpu").item())
            else:
                preds.append(pred.reshape(-1))
                labels.append(label.reshape(-1))

        if train:
            final_loss = sum(batch_loss) / len(batch_loss)
        else:
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            acc = sum(preds == labels) / len(preds)
            f1 = f1_score(labels, preds)
        return final_loss, acc, f1

    def train(self, train_loader, num_epoch, valid_loader=None, test_loader=None):
        losses = []
        for epoch in tqdm(range(num_epoch)):
            loss, _, _ = self.epoch_loop(epoch, train_loader, train=True)
            losses.append(loss)
            log_sentence = f"Epoch{epoch}: loss: {loss:.4f}"
            if valid_loader is not None:
                _, valid_acc, valid_f1 = self.epoch_loop(epoch, valid_loader, train=False)
                log_sentence += f" val_acc: {valid_acc:.4f} val_f1: {valid_f1:.4f}"
            if test_loader is not None:
                _, test_acc, test_f1 = self.epoch_loop(epoch, test_loader, train=False)
                log_sentence += f" test_acc: {test_acc:.4f} test_f1: {test_f1:.4f}"
            tqdm.write(log_sentence)
            if epoch == self.mid_output_epoch:
                out = self.output_path.split(".")[0]
                torch.save(self.model.state_dict(), f"{out}_epoch{self.mid_output_epoch}.pth")

        torch.save(self.model.state_dict(), self.output_path)