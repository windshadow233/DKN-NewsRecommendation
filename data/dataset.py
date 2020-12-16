import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from data.utils import *

# os.chdir('..')
import json
import time
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
                "clicked_news": [
                    {
                        "title": num_words_per_news,
                        "entities": num_words_per_news
                    } * num_clicked_news_per_user
                ]
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
        for i, one_clicked in enumerate(clicked):
            clicked_titles = one_clicked['title']
            clicked_entities = one_clicked['entities']
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
    def __init__(self, word_dict):
        with open(word_dict, 'r') as f:
            self.char2idx = json.loads(f.read())
        self.idx2char = dict(zip(self.char2idx.values(), self.char2idx.keys()))

    def wors2token(self, words):
        words = [self.char2idx.get(s, self.char2idx.get('<unk>'))
                 for s in words][:config.num_words_per_news]
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
                 behaviors='data/train/behaviors.tsv',
                 news='data/train/news.tsv',
                 positive_rate=0.3):
        self.title_converter = title_converter
        self.entity_converter = entity_converter
        print('Loading data...')
        self.behaviors = pd.read_csv(behaviors, sep='\t', header=None).set_index(0)
        self.behaviors.index.name = 'Impression_ID'
        self.behaviors.columns = ['User_ID', 'Time', 'History', 'Impressions']
        self.users_id = self.behaviors.User_ID.unique().tolist()
        # 先按User_ID分组,以加快读取速度
        self.behaviors = self.behaviors.groupby(by='User_ID')
        self.news = pd.read_csv(news, sep='\t', header=None, index_col=0)
        self.news.index.name = 'News_ID'
        self.news.columns = ['Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities',
                             'Abstract_Entities']
        self.positive_rate = positive_rate
        print('Finish!')

    def __getitem__(self, item):

        """
        Returns:
            {
                "candidate_news": {
                    "title": num_words_per_news,
                    "entities": num_words_per_news,
                    "is_click": 1
                },
                "clicked_news": [
                    {
                        "title": num_words_per_news,
                        "entities": num_words_per_news
                    } * num_clicked_news_per_user
                ]
            }
        """
        user_id = self.users_id[item]
        behaviors = self.behaviors.get_group(user_id)
        click_history = []
        impressions = []
        for i, behavior in behaviors.iterrows():
            history = behavior.History
            if not pd.isna(history):
                click_history.extend(behavior.History.split(' '))
            impressions.extend(behavior.Impressions.split(' '))
        ####################### candidate news #######################
        # 以positive_rate概率抽取正例
        positive = list(filter(lambda x: x[-1] == '1', impressions))
        if random.random() < self.positive_rate and positive:
            candidate_id, is_click = random.choice(positive).split('-')
        else:
            candidate_id, is_click = random.choice(impressions).split('-')
        # 直接随机抽取
        # candidate_id, is_click = random.choice(impressions).split('-')
        is_click = torch.tensor(int(is_click), dtype=torch.float32)
        candidate = self.news.loc[candidate_id]
        split_title = split_words(candidate.Title)
        candidate_title = self.title_converter.wors2token(split_title)
        entities = candidate.Title_Entities
        entities = [] if pd.isna(entities) else json.loads(entities)
        entities = get_entities_from_title(split_title, entities)
        candidate_title_entities = self.entity_converter.wors2token(entities)
        ######################## clicked news ########################
        clicked_news = []
        for history in click_history[:config.num_clicked_news_per_user]:
            to_add = {}
            news = self.news.loc[history]
            split_title = split_words(news.Title)
            entities = news.Title_Entities
            entities = [] if pd.isna(entities) else json.loads(entities)
            entities = get_entities_from_title(split_title, entities)
            to_add['title'] = self.title_converter.wors2token(split_title)
            to_add['entities'] = self.entity_converter.wors2token(entities)
            clicked_news.append(to_add)
        # 若clicked_news不够num_clicked_news_per_user条,补0(待考虑)
        pad_vec = torch.zeros(config.num_words_per_news, dtype=torch.long)
        clicked_news.extend([{'title': pad_vec, 'entities': pad_vec}
                             for _ in range(config.num_clicked_news_per_user - len(clicked_news))])
        to_return = {
            'candidate_news': {
                'title': candidate_title,
                'entities': candidate_title_entities,
                'is_click': is_click
            },
            'clicked_news': clicked_news
        }
        return to_return

    def __len__(self):
        return len(self.users_id)
