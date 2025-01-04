from bs4 import BeautifulSoup, NavigableString

TAG_ATTRIBUTE_MAP = {
    "d": ("certainty", "d"),
    "a": ("None", "a"),
    "f": ("None", "f"),
    "c": ("None", "c"),
    "timex3": ("type", "timex3"),
    "t-test": ("state", "t-test"),
    "t-key": ("None", "t-key"),
    "t-val": ("None", "t-val"),
    "m-key": ("state", "m-key"),
    "m-val": ("None", "m-val"),
    "r": ("state", "r"),
    "cc": ("state", "cc"),
    "p": ("None", "p"),
}

LABEL_NAME = {
    "d": "disease",
    "a": "anatomical-parts",
    "f": "feature",
    "c": "change",
    "timex3": "time",
    "t-test": "test-test",
    "t-key": "test-key",
    "t-val": "test-value",
    "m-key": "medicine-key",
    "m-val": "medicine-value",
    "r": "remedy",
    "cc": "clinical-context",
    "p": "pending",
}


def parse_node(node, current_label="O", current_detail="O", results=[]):
    tag_name = node.name

    if tag_name in ["articles", "article"]:
        for child in node.children:
            parse_node(child, current_label, current_detail, results)
        return results

    if isinstance(node, NavigableString):
        text = node.strip()
        if text:
            tokens = text.split()
            for i, t in enumerate(tokens):
                if current_label == "O":
                    results.append((t, "O", "O"))
                else:
                    if i == 0:
                        results.append((t, "B-" + current_detail, "B-" + LABEL_NAME[current_label]))
                    else:
                        results.append((t, "I-" + current_detail, "I-" + LABEL_NAME[current_label]))

        return results

    else:
        attr_key, current_label = TAG_ATTRIBUTE_MAP[tag_name]
        current_detail = node.get(attr_key, "O")

    for child in node.children:
        parse_node(child, current_label, current_detail, results)

    return results


def convert_xml_to_conll(xml_str):
    xml_str = xml_str.replace(".", " .")
    xml_str = xml_str.replace(",", " ,")

    soup = BeautifulSoup(xml_str, "xml")
    articles = soup.find_all("article")
    all_results = []
    for article in articles:
        results = []
        parse_node(article, "O", "O", results)
        all_results.append(results)

    return all_results


if __name__ == "__main__":

    with open("./data/MedTxt-CR-EN-training-pub.xml", encoding="utf-8") as f:
        original_data = f.read()
    article_results = convert_xml_to_conll(original_data)

    train_rate = 0.8
    train_num = int(len(article_results) * train_rate)
    train_text = ""
    for article_result in article_results[:train_num]:
        train_text += "-DOCSTART- -X- O\n\n"
        for word, detail, label in article_result:
            train_text += f"{word} {detail} {label}\n"
            if word == ".":
                train_text += "\n"

    test_text = ""
    for article_result in article_results[train_num:]:
        test_text += "-DOCSTART- -X- O\n\n"
        for word, detail, label in article_result:
            test_text += f"{word} {detail} {label}\n"
            if word == ".":
                test_text += "\n"

    with open("./data/MedTxt_train.txt", "w", encoding="utf-8") as f_out:
        f_out.write(train_text)

    with open("./data/MedTxt_test.txt", "w", encoding="utf-8") as f_out:
        f_out.write(test_text)
