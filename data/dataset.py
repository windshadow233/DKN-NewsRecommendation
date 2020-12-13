import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from data.utils import *

# os.chdir('..')
import json
import nltk
from data.utils import *
from config import config
device = torch.device(config.device)


def user_data_collate(one_batch):
    """
    Args:
        one_batch:
            ({
                "candidate_news": {
                    "title": num_words_per_news,
                    "entities": num_words_per_news,
                    "is_click": 1
                },
                "clicked_news": {
                    "titles": [num_words_per_news] * num_clicked_news_per_user,
                    "entities": [num_words_per_news] * num_clicked_news_per_user
                }
            } * batch_size)
    Return:
        candidate_news:
            {
                "titles": batch_size * num_words_per_news,
                "entities": batch_size * num_words_per_news
            }
        clicked_news:
            [
                {
                    "titles": batch_size * num_words_per_news,
                    "entities": batch_size * num_words_per_news
                } * num_clicked_news_per_user
            ]
        is_click: batch_size
    """
    clicked_news_titles = [[] for _ in range(config.num_clicked_news_per_user)]
    clicked_news_entities = [[] for _ in range(config.num_clicked_news_per_user)]
    candidate_news_titles = []
    candidate_news_entities = []
    is_click = []
    for data in one_batch:
        candidate = data['candidate_news']
        clicked = data['clicked_news']
        candidate_news_titles.append(candidate['title'])
        candidate_news_entities.append(candidate['entities'])
        is_click.append(candidate['is_click'])
        for i, (clicked_titles, clicked_entities) in enumerate(zip(clicked['titles'], clicked['entities'])):
            clicked_news_titles[i].append(clicked_titles)
            clicked_news_entities[i].append(clicked_entities)
    clicked_news_titles = list(map(lambda x: torch.stack(x), clicked_news_titles))
    clicked_news_entities = list(map(lambda x: torch.stack(x), clicked_news_entities))
    candidate_news = {
        'titles': torch.stack(candidate_news_titles).to(device),
        'entities': torch.stack(candidate_news_entities).to(device)
    }
    clicked_news = [{'titles': titles.to(device), 'entities': entities.to(device)}
                    for titles, entities in zip(clicked_news_titles, clicked_news_entities)]
    is_click = torch.stack(is_click).to(device)
    return candidate_news, clicked_news, is_click


class WordsTokenConverter:
    def __init__(self, word_dict='data/title_words_vocab.json'):
        with open(word_dict, 'r') as f:
            self.char2idx = json.loads(f.read())
        self.idx2char = dict(zip(self.char2idx.values(), self.char2idx.keys()))

    def wors2token(self, words):
        words = [self.char2idx.get(normalize_string(s), self.char2idx.get('<unk>'))
                 for s in nltk.word_tokenize(words)][:config.num_words_per_news]
        words.extend([self.char2idx.get('<pad>')] * (config.num_words_per_news - len(words)))
        return torch.tensor(words, dtype=torch.long)

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
        self.users_id = self.behaviors.User_ID.unique().to_list()

    def __getitem__(self, item):

        """
        Returns:
            {
                "candidate_news": {
                    "title": num_words_per_news,
                    "entities": num_words_per_news,
                    "is_click": 1
                },
                "clicked_news": {
                    "titles": [num_words_per_news * num_clicked_news_per_user],
                    "entities": [num_words_per_news * num_clicked_news_per_user]
                }
            }
        """

        user_id = self.users_id[item]
        behaviors = self.behaviors[self.behaviors.User_ID == user_id]
        click_history = []
        impressions = []
        titles = []
        title_entities = []
        for i, behavior in behaviors.iterrows():
            history = behavior.History
            if not pd.isna(history):
                click_history.extend(behavior.History.split(' '))
            impressions.extend(behavior.Impressions.split(' '))
        # candidate news
        candidate_id, is_click = random.choice(impressions).split('-')
        is_click = torch.tensor(int(is_click), dtype=torch.float32)
        candidate = self.news.loc[candidate_id]
        candidate_title = self.title_converter.wors2token(candidate.Title)
        entities = candidate.Title_Entities
        entities = [] if pd.isna(entities) else json.loads(entities)
        candidate_title_entities = self.entity_converter.wors2token(' '.join([
            entity.get('WikidataId') for entity in entities
        ]))
        # clicked_news
        for history in click_history[:config.num_clicked_news_per_user]:
            news = self.news.loc[history]
            titles.append(self.title_converter.wors2token(news.Title))
            entities = news.Title_Entities
            entities = [] if pd.isna(entities) else json.loads(entities)
            title_entities.append(self.entity_converter.wors2token(' '.join([
                entity.get('WikidataId') for entity in entities
            ])))
        # 若clicked_news不够num_clicked_news_per_user条,补0(待考虑)
        titles.extend([torch.zeros(size=(config.num_words_per_news,), dtype=torch.long)]
                      * (config.num_clicked_news_per_user - len(titles)))
        title_entities.extend([torch.zeros(size=(config.num_words_per_news,), dtype=torch.long)]
                              * (config.num_clicked_news_per_user - len(title_entities)))
        to_return = {
            'candidate_news': {
                'title': candidate_title,
                'entities': candidate_title_entities,
                'is_click': is_click
            },
            'clicked_news': {
                'titles': titles,
                'entities': title_entities
            }
        }
        return to_return

    def __len__(self):
        return len(self.users_id)
