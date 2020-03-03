# Arabic BERT Model

Pretrained BERT base language model for Arabic

## Pretraining Corpus

`bert-base-arabic` model was pretrained on ~8.2 Billion words:

- Arabic version of [OSCAR](https://traces1.inria.fr/oscar/) - filtered from [Common Crawl](http://commoncrawl.org/)
- Recent dump of Arabic [Wikipedia](https://dumps.wikimedia.org/backup-index.html)

and other Arabic resources which sum up to ~95GB of text.

__Notes on training data:__

- Our final version of corpus contains some non-Arabic words inlines, which we did not remove from sentences since that would affect some tasks like NER.
- Although non-Arabic characters were lowered as a preprocessing step, since Arabic characters does not have upper or lower case, there is no cased and uncased version of the model.
- The corpus and vocabulary set are not restricted to Modern Standard Arabic, they contain some dialectical Arabic too.

## Pretraining details

- This model was trained using Google BERT's github [repository](https://github.com/google-research/bert) on a single TPU v3-8 provided for free from [TFRC](https://www.tensorflow.org/tfrc).
- Our pretraining procedure follows training settings of bert with some changes: trained for 3M training steps with batchsize of 128, instead of 1M with batchsize of 256.
- You can find the outputs of the training process on tensorboard: [Arabic-BERT](https://tensorboard.dev/experiment/qv6llNexQKeXsBoT9ooRoQ/)

## Load Pretrained Model

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
model = AutoModel.from_pretrained("asafaya/bert-base-arabic")
```

## Results


### Sentiment Analysis Results (F1-Score)

| Dataset   | Details | [ML-BERT](https://github.com/google-research/bert/blob/master/multilingual.md)   | [hULMona](https://github.com/aub-mind/hULMonA)  | Arabic-BERT  |
|:---------:|:-------:|:---------:|:--------:|:------------:|
| [HARD](https://github.com/elnagara/HARD-Arabic-Dataset) | 2 Classes, Mixed dialects | 0.957     | 0.957    | -            |
| [ArSenLev](https://arxiv.org/abs/1906.01830) | 5 Classes, Levantine dialect  | 0.510     | 0.511    | __0.552__    |
| [ASTD](https://www.sites.google.com/a/mohamedaly.info/www/datasets/astd) |  4 Classes, MSA and Egyptian dialects | 0.670     | 0.677    | __0.714__    |



### Named Entity Recognition

To be added.

__Note:__ More results on other downstream NLP tasks will be added soon. if you use this model, I would appreciate your feedback.

## Acknowledgement

Thanks to Google for providing free TPU for the training process and for Huggingface for hosting this model on their servers 😊
