from datasets import load_dataset


def get_dataset():
    dataset = load_dataset("glue", "sst2")  # dataset = load_dataset('glue', 'mrpc')
    train_datasets = dataset["train"]
    sentences = train_datasets["sentence"]
    labels = train_datasets["label"]
    out = ""
    for sentence, label in zip(sentences, labels):
        out += f"{sentence}\n{label}\n1.0\n\n"
    with open("./data/sst2_train.txt", "w", encoding="utf-8") as f:
        f.write(out)

    valid_datasets = dataset["validation"]
    sentences = valid_datasets["sentence"]
    labels = valid_datasets["label"]
    out = ""
    for sentence, label in zip(sentences, labels):
        out += f"{sentence}\n{label}\n1.0\n\n"
    with open("./data/sst2_valid.txt", "w", encoding="utf-8") as f:
        f.write(out)

    test_datasets = dataset["test"]
    sentences = test_datasets["sentence"]
    labels = test_datasets["label"]
    out = ""
    for sentence, label in zip(sentences, labels):
        out += f"{sentence}\n{label}\n1.0\n\n"
    with open("./data/sst2_test_masked.txt", "w", encoding="utf-8") as f:
        f.write(out)

    test_datasets2 = load_dataset("gpt3mix/sst2")["test"]
    sentences2 = test_datasets2["text"]
    labels2 = test_datasets2["label"]
    out = ""
    for sentence, label in zip(sentences2, labels2):
        sentence = (
            sentence.lower()
            .replace("ﾃｩ", "é")
            .replace("ﾃｭ", "í")
            .replace("ﾃｼ", "ü")
            .replace("ﾃｳ", "ó")
            .replace("ﾃｨ", "è")
            .replace("ﾃｻ", "û")
            .replace("\\/", "/")
            .replace("\\*", "*")
            .replace("-lrb-", "(")
            .replace("-rrb-", ")")
            .replace("no. .", "no . .")
            .replace("wanna ", "wan na ")
            # .replace("wannabe", "wan nabe")
            .replace("learnt", "learned")
            .replace("favour", "favor")
            .replace("favourite", "favorite")
            .replace("humour ", "humor ")
        )
        label = 1 if label == 0 else 0
        if sentence not in sentences:
            print(sentence, label)
        out += f"{sentence}\n{label}\n1.0\n\n"
    with open("./data/sst2_test.txt", "w", encoding="utf-8") as f:
        f.write(out)

    dataset = load_dataset("SetFit/mrpc")
    train_datasets = dataset["train"]
    sentences1 = train_datasets["text1"]
    sentences2 = train_datasets["text2"]
    labels = train_datasets["label"]
    out = ""
    for sentence1, sentence2, label in zip(sentences1, sentences2, labels):
        out += f"{sentence1}\n{sentence2}\n{label}\n1.0\n\n"
    with open("./data/mrpc_train.txt", "w", encoding="utf-8") as f:
        f.write(out)

    valid_datasets = dataset["validation"]
    sentences1 = valid_datasets["text1"]
    sentences2 = valid_datasets["text2"]
    labels = valid_datasets["label"]
    out = ""
    for sentence1, sentence2, label in zip(sentences1, sentences2, labels):
        out += f"{sentence1}\n{sentence2}\n{label}\n1.0\n\n"
    with open("./data/mrpc_valid.txt", "w", encoding="utf-8") as f:
        f.write(out)

    test_datasets = dataset["test"]
    sentences1 = test_datasets["text1"]
    sentences2 = test_datasets["text2"]
    labels = test_datasets["label"]
    out = ""
    for sentence1, sentence2, label in zip(sentences1, sentences2, labels):
        out += f"{sentence1}\n{sentence2}\n{label}\n1.0\n\n"
    with open("./data/mrpc_test.txt", "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    get_dataset()
