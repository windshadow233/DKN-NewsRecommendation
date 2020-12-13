import pandas as pd
import pickle
import json
import nltk
from collections import Counter
import os
import numpy as np
import tqdm
from data.utils import *

behaviors = pd.read_csv('small_train/behaviors.tsv', sep='\t', header=None)
behaviors.index.name = 'Impression_ID'
behaviors.columns = ['User_ID', 'Time', 'History', 'Impressions']
news = pd.read_csv('small_train/news.tsv', sep='\t', header=None, index_col=0)
news.index.name = 'News_ID'
news.columns = ['Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']

if not os.path.exists('title_entities_vocab.json'):
    entities = ['<pad>', '<unk>']
    with open("small_train/entity_embedding.vec", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            entities.append(line[0])
    entities = dict(zip(entities, range(len(entities))))
    with open('title_entities_vocab.json', 'w') as f:
        f.write(json.dumps(entities))

if not os.path.exists('title_entities_embedding.pkl'):
    with open("small_train/entity_embedding.vec", "r") as f:
        vec = []
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            vec.append(np.array([float(item) for item in line[1:]]))
        vec.insert(0, np.zeros(shape=(len(vec[-1]))))
        vec.insert(0, np.zeros(shape=(len(vec[-1]))))
        vec = np.stack(vec)

    with open('title_entities_embedding.pkl', 'wb') as f:
        f.write(pickle.dumps(vec))

if not os.path.exists('title_words_vocab.json'):
    word_freq = Counter()
    for title in tqdm.tqdm(news.Title):
        title = title.lower()
        tokens = [normalize_string(s) for s in nltk.word_tokenize(title)]
        word_freq.update(tokens)
    cumsum = np.cumsum(sorted(list(word_freq.values()), reverse=True))
    total = cumsum[-1] * 0.97
    vocab_size = list(cumsum > total).index(True)
    words = word_freq.most_common(vocab_size)
    word_map = {'<pad>': 0, '<unk>': 1}
    word_map.update({k[0]: v + 2 for v, k in enumerate(words)})
    with open('title_words_vocab.json', 'w') as f:
        f.write(json.dumps(word_map))
