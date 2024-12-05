import random

from transformers import RobertaTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_pairs(word, p=0):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        if p < random.random() or prev_char == "Ġ":
            pairs.add((prev_char, char))
        else:
            pairs.add((prev_char, "@@" + char))
        prev_char = char
    return pairs


class RobertaTokenizerDropout(RobertaTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        p=0,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self.p = p
        self._p = p

    def const_tokenize(self):
        self.p = 0

    def random_tokenize(self):
        self.p = self._p

    def bpe(self, token):
        if (token in self.cache) and (self.p == 0):
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word, self.p)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            if len(new_word) == 1:
                word = new_word
                break

            word = new_word
            pairs = get_pairs(word, self.p)

        word = " ".join(word)
        if self.p == 0:
            self.cache[token] = word
        return word

    def tokenizeSentence(self, text):
        text = text.split(" ")
        subwords = []
        word_ids = [-1]
        for i, token in enumerate(text):
            if i == 0:
                subword = self.tokenize(token)
            else:
                subword = self.tokenize(" " + token)
            subwords += subword
            word_ids += [max(word_ids) + 1] * len(subword)

        return subwords, word_ids[1:]


if __name__ == "__main__":
    tokenizer = RobertaTokenizerDropout.from_pretrained("roberta-large", p=0.1)
    text = "This is an apple."
    for i in range(10):
        print(tokenizer.tokenizeSentence(text))
