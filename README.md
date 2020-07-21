# Arabic BERT

Pretrained BERT language models for Arabic

_If you use any of these models in your work, please cite this paper (to appear in SemEval2020 proceedings):_

```
@inproceedings{
  title={KUISAIL at SemEval-2020 Task 12: BERT-CNN for Offensive Speech Identification in Social Media},
  author={Safaya, Ali and Abdullatif, Moutasem and Yuret, Deniz},
  booktitle={Proceedings of the International Workshop on Semantic Evaluation (SemEval)},
  year={2020}
}
```

## Pretraining data

The models were pretrained on ~8.2 Billion words:

- Arabic version of [OSCAR](https://oscar-corpus.com/) (unshuffled version of the corpus) - filtered from [Common Crawl](http://commoncrawl.org/)
- Recent dump of Arabic [Wikipedia](https://dumps.wikimedia.org/backup-index.html)

and other Arabic resources which sum up to ~95GB of text.

__Notes on training data:__

- Our final version of corpus contains some non-Arabic words inlines, which we did not remove from sentences since that would affect some tasks like NER.
- Although non-Arabic characters were lowered as a preprocessing step, since Arabic characters do not have upper or lower case, there is no cased and uncased version of the model.
- The corpus and vocabulary set are not restricted to Modern Standard Arabic, they contain some dialectical Arabic too.

## Pretraining details

- These models were trained using Google BERT's github [repository](https://github.com/google-research/bert) on a single TPU v3-8 provided for free from [TFRC](https://www.tensorflow.org/tfrc).
- Our pretraining procedure follows training settings of bert with some changes: trained for 3M training steps with batchsize of 128, instead of 1M with batchsize of 256.

## Models

|  | BERT-Mini | BERT-Medium   | BERT-Base  | BERT-Large  |
|:---:|:---:|:---:|:---:|:---:|
| Hidden Layers | 4 | 8 | 12 | 24 |
| Attention heads | 4 | 8 | 12 | 16 |
| Hidden size | 256 | 512 | 768 | 1024 |
| Parameters | 11M | 42M | 110M | 340M |

## Results


### Sentiment Analysis Results (F1-Score)

| Dataset   | Details | [ML-BERT](https://github.com/google-research/bert/blob/master/multilingual.md)   | [hULMona](https://github.com/aub-mind/hULMonA)  | Arabic-BERT Base  |
|:---------:|:-------:|:---------:|:--------:|:------------:|
| [HARD](https://github.com/elnagara/HARD-Arabic-Dataset) | 2 Classes, Mixed dialects | 0.957     | 0.957    | -            |
| [ArSenLev](https://arxiv.org/abs/1906.01830) | 5 Classes, Levantine dialect  | 0.510     | 0.511    | __0.552__    |
| [ASTD](https://www.sites.google.com/a/mohamedaly.info/www/datasets/astd) |  4 Classes, MSA and Egyptian dialects | 0.670     | 0.677    | __0.714__    |


__Note:__ More results on other downstream NLP tasks will be added soon. if you use these models, I would appreciate your feedback.

## How to use

You can use these models by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import AutoTokenizer, AutoModel

# Mini:   asafaya/bert-mini-arabic
# Medium: asafaya/bert-medium-arabic
# Base:   asafaya/bert-base-arabic
# Large:  asafaya/bert-large-arabic

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
model = AutoModel.from_pretrained("asafaya/bert-base-arabic")
```

## Acknowledgement

Thanks to Google for providing free TPU for the training process and for Huggingface for hosting these models on their servers 😊
