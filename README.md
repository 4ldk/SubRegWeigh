# SubRegWeigh

This repository is the official implementation of the COLING 2025 paper:
**[SubRegWeigh: Effective and Efficient AnnotationWeighing with Subword Regularization](https://aclanthology.org/2025.coling-main.130/)**

## Datasets
Put following data on ./data folder 
- CoNLL 2003 original: https://www.clips.uantwerpen.be/conll2003/ner/
- CoNLL++(CoNLL CW): https://github.com/ZihanWangKi/CrossWeigh
- CoNLL++(CoNLL 2020): https://github.com/ShuhengL/acl2023_conllpp
- SST2: https://huggingface.co/datasets/stanfordnlp/sst2

## Create Weighted Dataset
`./SubRegWeigh/scripts/run.sh`

## Results
*Note*: For detailed experimental results, please refer to our paper.

| Models                | Time | CoNLL CW | CoNLL 2020 |
|-----------------------|------|----------|------------|
| RoBERTA LARGE         |     -|     95.27|       94.80|
| +SubRegWeigh (K-means)|  5:21|     95.45|       94.96|
| LUKE LARGE            |     -|     95.92|       95.29|
| +SubRegWeigh (K-means)|  6:36|     96.12|       95.31| 

## Citation

```bibtex
@inproceedings{tsuji-etal-2025-subregweigh,
    title = "{S}ub{R}eg{W}eigh: Effective and Efficient Annotation Weighing with Subword Regularization",
    author = "Tsuji, Kohei  and
      Hiraoka, Tatsuya  and
      Cheng, Yuchang  and
      Iwakura, Tomoya",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.130/",
    pages = "1908--1921"
}
```
