import os
import sys

import torch
from transformers import AutoConfig, RobertaForTokenClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from roberta.bpe_dropout import RobertaTokenizerDropout


class RoBERTaWithNoise(RobertaForTokenClassification):
    def forward(self, input_ids=None, attention_mask=None, alpha=0.1, *args, **kwargs):
        inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        noise = torch.randn_like(inputs_embeds) * alpha
        noisy_inputs_embeds = inputs_embeds + noise

        return super().forward(inputs_embeds=noisy_inputs_embeds, attention_mask=attention_mask, *args, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(42)

    model_name = "roberta-large"
    tokenizer = RobertaTokenizerDropout.from_pretrained(model_name, p=0)
    config = AutoConfig.from_pretrained(model_name, num_labels=9)
    model = RoBERTaWithNoise.from_pretrained(model_name)

    text = "This is an apple."
    inputs = tokenizer(text, return_tensors="pt")
    out = model(**inputs, alpha=1).logits
    print(out[0, 0])
