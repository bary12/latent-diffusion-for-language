from datasets import load_dataset
import nltk
from huggingface_hub import login
from pathlib import Path
import os

from functools import partial
from datasets import Dataset

nltk.download('punkt')
nltk.download('punkt_tab')


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


c4 = load_dataset("allenai/c4", "en", split='train', streaming=True)


def sent_tokenize(batch):
    return {'text': [sent for text in batch['text'] for sent in nltk.sent_tokenize(text)]}


c4_100k = c4.take(100000)

c4_100k = Dataset.from_generator(partial(gen_from_iterable_dataset, c4_100k), features=c4_100k.features)

c4_sentences = c4_100k.map(sent_tokenize, batched=True,
                           remove_columns=c4_100k.column_names)

c4_sentences = c4_sentences.shuffle()

print(c4_sentences)

login(token=os.getenv('HUGGINGFACE_TOKEN'))

c4_sentences.push_to_hub('Bary/c4-sentences-2M')
