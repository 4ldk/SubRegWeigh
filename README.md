# SubRegWeigh

This repository is the official implementation of the COLING 2025 paper:
**[SubRegWeigh: Effective and Efficient AnnotationWeighing with Subword Regularization](https://arxiv.org/abs/2409.06216)**

## Datasets
Put following data on ./data folder 
- CoNLL 2003 original: https://www.clips.uantwerpen.be/conll2003/ner/
- CoNLL++(CoNLL CW): https://github.com/ZihanWangKi/CrossWeigh
- CoNLL++(CoNLL 2020): https://github.com/ShuhengL/acl2023_conllpp
- SSL2: https://huggingface.co/datasets/stanfordnlp/sst2

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
*Note*: This citation refers to the preprint version on arXiv. The official COLING 2025 proceedings citation will be updated once it becomes available.

```bibtex
@article{tsuji2024subregweigh,
  title={SubRegWeigh: Effective and Efficient Annotation Weighing with Subword Regularization},
  author={Tsuji, Kohei and Hiraoka, Tatsuya and Cheng, Yuchang and Iwakura, Tomoya},
  journal={arXiv preprint arXiv:2409.06216},
  year={2024}
}
```
