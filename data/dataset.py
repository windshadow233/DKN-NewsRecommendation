import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from data.utils import *

# os.chdir('..')
import json
import nltk
from data.utils import *
from config import config


class WordsTokenConverter:
    def __init__(self, word_dict='data/title_words_vocab.json'):
        with open(word_dict, 'r') as f:
            self.char2idx = json.loads(f.read())
        self.idx2char = dict(zip(self.char2idx.values(), self.char2idx.keys()))

    def wors2token(self, words):
        words = [self.char2idx.get(normalize_string(s), self.char2idx.get('<unk>'))
                 for s in nltk.word_tokenize(words)][:config.num_words_per_news]
        words.extend([self.char2idx.get('<pad>')] * (config.num_words_per_news - len(words)))
        return words

    def token2words(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ' '.join([self.idx2char.get(token) for token in tokens if token != config.pad_idx])


class UserDataset(Dataset):
    def __init__(self,
                 title_converter: WordsTokenConverter,
                 entity_converter: WordsTokenConverter,
                 behaviors='data/small_train/behaviors.tsv',
                 news='data/small_train/news.tsv'):
        self.title_converter = title_converter
        self.entity_converter = entity_converter
        self.behaviors = pd.read_csv(behaviors, sep='\t', header=None)
        self.behaviors.index.name = 'Impression_ID'
        self.behaviors.columns = ['User_ID', 'Time', 'History', 'Impressions']
        self.news = pd.read_csv(news, sep='\t', header=None, index_col=0)
        self.news.index.name = 'News_ID'
        self.news.columns = ['Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities',
                             'Abstract_Entities']

    def __getitem__(self, item):
        """
        Returns:
            {
                "titles": [[num_words_per_news] * num_news]
                "entities": [[num_words_per_news] * num_news]
            }
        """
        behavior = self.behaviors.iloc[item]
        user_id = behavior.User_ID
        history = behavior.History.split(' ')
        impressions = behavior.Impressions.split(' ')
        titles = []
        title_entities = []
        for news_id in history:
            titles.append(self.news.loc[news_id].Title)
            title_entities.append(self.entity_converter.wors2token(
                ' '.join([item.get('WikidataId') for item in json.loads(self.news.loc[news_id].Title_Entities)])))
        tokens = []
        for title in titles:
            token = self.title_converter.wors2token(title)
            tokens.append(token)
        return {'titles': tokens, 'entities': title_entities}

    def __len__(self):
        return len(self.behaviors)
