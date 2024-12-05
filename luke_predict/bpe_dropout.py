import copy
import random

from transformers import LukeTokenizer


def get_pairs(word, p=0):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        if p < random.random() or prev_char == "Ġ":
            pairs.add((prev_char, char))
        else:
            pairs.add((prev_char, "@@" + char))
        prev_char = char
    return pairs


class LukeTokenizerDropout(LukeTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        entity_vocab_file,
        task=None,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        entity_unk_token="[UNK]",
        entity_pad_token="[PAD]",
        entity_mask_token="[MASK]",
        entity_mask2_token="[MASK2]",
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
        seed=42,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            entity_vocab_file,
            task,
            max_entity_length,
            max_mention_length,
            entity_token_1,
            entity_token_2,
            entity_unk_token,
            entity_pad_token,
            entity_mask_token,
            entity_mask2_token,
            errors,
            bos_token,
            eos_token,
            sep_token,
            cls_token,
            unk_token,
            pad_token,
            mask_token,
            add_prefix_space,
            **kwargs,
        )
        self.p = p
        self._p = p
        self.seed = seed
        random.seed(self.seed)

    def const_tokenize(self):
        self._p = copy.deepcopy(self.p)
        self.p = 0

    def random_tokenize(self):
        self.p = self._p

    def reset_seed(self):
        random.seed(self.seed)

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
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word, self.p)
        word = " ".join(word)
        if self.p == 0:
            self.cache[token] = word
        return word
