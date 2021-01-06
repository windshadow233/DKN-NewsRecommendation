import pandas as pd
import pickle
import json
from collections import Counter
import os
import numpy as np
import tqdm
from data.utils import *

print('Loading Data')
news = pd.read_csv('train/news.tsv', sep='\t', header=None, index_col=0)
news.index.name = 'News_ID'
news.columns = ['Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']
behaviors = pd.read_csv('train/behaviors.tsv', sep='\t', header=None).set_index(0)
behaviors.index.name = 'Impression_ID'
behaviors.columns = ['User_ID', 'Time', 'History', 'Impressions']
print('Finish!')

"""生成子类词典"""
if not os.path.exists('subcategories.json'):
    with open('subcategories.json', 'w') as f:
        subcategory = news.SubCategory.unique()
        f.write(json.dumps(dict(zip(subcategory, range(1, len(subcategory) + 1)))))

"""生成类别词典"""
if not os.path.exists('categories.json'):
    with open('categories.json', 'w') as f:
        category = news.Category.unique()
        f.write(json.dumps(dict(zip(category, range(1, len(category) + 1)))))

"""生成标题的实体嵌入文件与标题实体词典"""
if not os.path.exists('entities_embedding.pkl'):
    with open("train/entity_embedding.vec", "r") as f,\
            open("test/entity_embedding.vec", "r") as g,\
            open("dev/entity_embedding.vec", "r") as h:
        d = {'<pad>': np.zeros(shape=(100,)),
             '<unk>': np.zeros(shape=(100,))}
        for file in [f, g, h]:
            lines = file.readlines()
            for line in tqdm.tqdm(lines):
                ID, *vec = line.strip().split("\t")
                d[ID] = np.array(vec).astype(float)
        entities = []
        vecs = []
        for ID, vec in tqdm.tqdm(d.items()):
            entities.append(ID)
            vecs.append(vec)
        vecs = np.stack(vecs)

    with open('entities_embedding.pkl', 'wb') as f:
        f.write(pickle.dumps(vecs))
    entities = dict(zip(entities, range(len(entities))))
    with open('entities_vocab.json', 'w') as f:
        f.write(json.dumps(entities))

"""生成标题词典"""
if not os.path.exists('title_words_vocab.json'):
    word_freq = Counter()
    for title in tqdm.tqdm(news.Title):
        title = title.lower()
        tokens = split_words(title)
        word_freq.update(tokens)
    cumsum = np.cumsum(sorted(list(word_freq.values()), reverse=True))
    total = cumsum[-1] * 0.97
    vocab_size = list(cumsum > total).index(True)
    words = word_freq.most_common(vocab_size)
    word_map = {'<pad>': 0, '<unk>': 1}
    word_map.update({k[0]: v + 2 for v, k in enumerate(words)})
    with open('title_words_vocab.json', 'w') as f:
        f.write(json.dumps(word_map))

"""统计新闻点击数并排序"""
if not os.path.exists('news_clicked_freq.json'):
    news_freq = Counter()
    for history, impressions in tqdm.tqdm(zip(behaviors.History, behaviors.Impressions), total=len(behaviors)):
        if not pd.isna(history):
            history = history.split(' ')
            news_freq.update(history)
        impressions = impressions.split(' ')
        clicked = [impression.split('-')[0] for impression in impressions if impression.split('-')[1] == '1']
        news_freq.update(clicked)
    with open('news_clicked_freq.json', 'w') as f:
        f.write(json.dumps(news_freq))
