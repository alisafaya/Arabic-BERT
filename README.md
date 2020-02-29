# Arabic-BERT

Pretrained BERT base language model for Arabic

## Pretraining Corpus

`arabic-bert-base` model was pretrained on ~8.2 Billion words:

- Arabic version of [OSCAR](https://traces1.inria.fr/oscar/) - filtered from [Common Crawl](http://commoncrawl.org/)
- Recent dump of Arabic [Wikipedia](https://dumps.wikimedia.org/backup-index.html)

and other Arabic resources which sum up to ~95GB of text.

__Notes on training data__: 

- Our final version of corpus contains some non-Arabic words inlines, which we did not remove from sentences since that would affect some tasks like NER.
- Although non-Arabic characters were lowered as a preprocessing step. Since Arabic characters does not have upper or lower case, there is no cased and uncased version of the model.
- The corpus and vocabulary set are not restricted to Modern Standard Arabic, they contain some dialectical Arabic too.


## Pretraining details

- This model was trained using Google BERT's github [repository](https://github.com/google-research/bert) on a single TPU v3-8 provided for free from [TFRC](https://www.tensorflow.org/tfrc), thanks for your support Google!.
- Our pretraining procedure follows training settings of bert with some changes: trained for 3M training steps with batchsize of 128, instead of 1M with batchsize of 256.


## Load Pretrained Model 

You can use this model by installing `torch` or `tensorflow` and Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("safaya/bert-base-arabic")
model = AutoModel.from_pretrained("safaya/bert-base-arabic")
```

## Acknowledgement

Thanks to Google for providing free TPU for the training process and for Huggingface for hosting this model on their servers.



