import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from data.utils import *

# os.chdir('..')
import json
import time
import tqdm
from data.utils import *
from config import config
from transformers import AutoTokenizer


device = torch.device(config.device)


def user_data_collate(one_batch):
    """
    Args:
        one_batch:
            ({
                "candidate_news": {
                    "title": num_words_per_news,
                    "entities": num_words_per_news,
                    "category": 1
                    "is_click": 1
                },
                "clicked_news": [
                    {
                        "title": num_words_per_news,
                        "entities": num_words_per_news,
                        "category": 1
                    } * num_clicked_news_per_user
                ]
            } * batch_size)
    Return:
        candidate_news:
            {
                "titles": batch_size * num_words_per_news,
                "entities": batch_size * num_words_per_news,
                "categories": batch_size
            }
        clicked_news:
            [
                {
                    "titles": batch_size * num_words_per_news,
                    "entities": batch_size * num_words_per_news,
                    "categories": batch_size
                } * num_clicked_news_per_user
            ]
        is_click: batch_size
    """
    clicked_news_titles = [[] for _ in range(config.num_clicked_news_per_user)]
    clicked_news_entities = [[] for _ in range(config.num_clicked_news_per_user)]
    clicked_news_categories = [[] for _ in range(config.num_clicked_news_per_user)]
    candidate_news_titles = []
    candidate_news_entities = []
    candidate_news_categories = []
    is_click = []
    for data in one_batch:
        candidate = data['candidate_news']
        clicked = data['clicked_news']
        candidate_news_titles.append(candidate['title'])
        candidate_news_entities.append(candidate['entities'])
        candidate_news_categories.append(candidate['category'])
        is_click.append(candidate['is_click'])
        for i, one_clicked in enumerate(clicked):
            clicked_titles = one_clicked['title']
            clicked_entities = one_clicked['entities']
            clicked_category = one_clicked['category']
            clicked_news_titles[i].append(clicked_titles)
            clicked_news_entities[i].append(clicked_entities)
            clicked_news_categories[i].append(clicked_category)
    clicked_news_titles = list(map(lambda x: torch.stack(x), clicked_news_titles))
    clicked_news_entities = list(map(lambda x: torch.stack(x), clicked_news_entities))
    clicked_news_categories = list(map(lambda x: torch.stack(x), clicked_news_categories))
    candidate_news = {
        'titles': torch.nn.utils.rnn.pad_sequence(candidate_news_titles, batch_first=True).to(device),
        'entities': torch.nn.utils.rnn.pad_sequence(candidate_news_entities, batch_first=True).to(device),
        'categories': torch.stack(candidate_news_categories).to(device)
    }
    clicked_news = [{'titles': titles.to(device),
                     'entities': entities.to(device),
                     'categories': categories.to(device)}
                    for titles, entities, categories
                    in zip(clicked_news_titles, clicked_news_entities, clicked_news_categories)]
    is_click = torch.stack(is_click).to(device)
    return candidate_news, clicked_news, is_click


class Dictionary:
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


class TrainDataset(Dataset):
    def __init__(self,
                 title_dict,
                 entity_dict,
                 positive_rate=0.5):
        """
        :param title_dict: 标题字典文件
        :param entity_dict: 实体字典文件
        :param positive_rate: 重采样正例概率
        """
        self.title_dict = Dictionary(title_dict)
        self.entity_dict = Dictionary(entity_dict)
        with open('data/categories.json', 'r') as f:
            self.category_dict = json.loads(f.read())
        print('Loading data...')
        self.behaviors = pd.read_csv('data/train/behaviors.tsv', sep='\t', header=None).set_index(0)
        self.behaviors.index.name = 'Impression_ID'
        self.behaviors.columns = ['User_ID', 'Time', 'History', 'Impressions']
        # 丢弃无价值数据
        self.behaviors.dropna(subset=['History'], inplace=True)
        self.users_id = self.behaviors.User_ID.unique().tolist()
        # 先按User_ID分组,以加快读取速度
        self.behaviors = self.behaviors.groupby(by='User_ID')
        self.news = pd.read_csv('data/train/news.tsv', sep='\t', header=None, index_col=0)
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
                    "category": 1
                    "is_click": 1
                },
                "clicked_news": [
                    {
                        "title": num_words_per_news,
                        "entities": num_words_per_news,
                        "category": 1
                    } * num_clicked_news_per_user
                ]
            }
        """
        user_id = self.users_id[item]
        behaviors = self.behaviors.get_group(user_id)
        impressions = []
        for i, (_, behavior) in enumerate(behaviors.iterrows()):
            if i == 0:
                history = behavior.History
                click_history = history.split(' ')
            impressions.extend(behavior.Impressions.split(' '))
        ####################### candidate news #######################
        # 以positive_rate概率抽取正例
        positive = list(filter(lambda x: x[-1] == '1', impressions))
        negative = list(filter(lambda x: x[-1] == '0', impressions))
        if random.random() < self.positive_rate and positive:
            candidate_id, is_click = random.choice(positive).split('-')
        else:
            candidate_id, is_click = random.choice(negative).split('-')
        # 直接随机抽取
        # candidate_id, is_click = random.choice(impressions).split('-')
        is_click = torch.tensor(int(is_click), dtype=torch.float32)
        candidate = self.news.loc[candidate_id]
        split_title = split_words(candidate.Title)
        candidate_category = torch.tensor(self.category_dict.get(candidate.Category, 0))
        candidate_title = self.title_dict.wors2token(split_title)
        entities = candidate.Title_Entities
        entities = [] if pd.isna(entities) else json.loads(entities)
        entities = get_entities_from_title(split_title, entities)
        candidate_title_entities = self.entity_dict.wors2token(entities)
        ######################## clicked news ########################
        clicked_news = []
        for history in click_history[-config.num_clicked_news_per_user:]:
            to_add = {}
            news = self.news.loc[history]
            split_title = split_words(news.Title)
            entities = news.Title_Entities
            entities = [] if pd.isna(entities) else json.loads(entities)
            entities = get_entities_from_title(split_title, entities)
            to_add['title'] = self.title_dict.wors2token(split_title)
            to_add['entities'] = self.entity_dict.wors2token(entities)
            to_add['category'] = torch.tensor(self.category_dict.get(news.Category, 0)).long()
            clicked_news.append(to_add)
        # 若clicked_news不够num_clicked_news_per_user条,补0(待考虑)
        pad_vec = torch.zeros(config.num_words_per_news, dtype=torch.long)
        clicked_news.extend([{'title': pad_vec, 'entities': pad_vec, 'category': torch.tensor(0)}
                             for _ in range(config.num_clicked_news_per_user - len(clicked_news))])
        to_return = {
            'candidate_news': {
                'title': candidate_title,
                'entities': candidate_title_entities,
                'is_click': is_click,
                'category': candidate_category
            },
            'clicked_news': clicked_news
        }
        return to_return

    def __len__(self):
        return len(self.users_id)


class TestDataset(object):
    def __init__(self,
                 title_dict,
                 entity_dict,
                 mode='test'):
        self.title_dict = Dictionary(title_dict)
        self.entity_dict = Dictionary(entity_dict)
        with open('data/categories.json', 'r') as f:
            self.category_dict = json.loads(f.read())
        print('Loading data...')
        # 无history的用户记录另作处理
        self.behaviors = pd.read_csv(f'data/{mode}/behaviors.tsv', sep='\t', header=None).set_index(0).dropna(subset=[3])
        self.behaviors.index.name = 'Impression_ID'
        self.behaviors.columns = ['User_ID', 'Time', 'History', 'Impressions']
        self.news = pd.read_csv(f'data/{mode}/news.tsv', sep='\t', header=None, index_col=0)
        self.news.index.name = 'News_ID'
        self.news.columns = ['Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'Title_Entities',
                             'Abstract_Entities']
        self.mode = mode
        print('Finish!')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, item):
        behavior = self.behaviors.iloc[item]
        history = behavior.History
        click_history = history.split(' ')
        impressions = behavior.Impressions.split(' ')
        clicked_news = []
        for history in click_history[-config.num_clicked_news_per_user:]:
            to_add = {}
            news = self.news.loc[history]
            split_title = split_words(news.Title)
            entities = news.Title_Entities
            entities = [] if pd.isna(entities) else json.loads(entities)
            entities = get_entities_from_title(split_title, entities)
            to_add['title'] = self.title_dict.wors2token(split_title)
            to_add['entities'] = self.entity_dict.wors2token(entities)
            to_add['category'] = torch.tensor(self.category_dict.get(news.Category, 0)).long()
            clicked_news.append(to_add)
        # 若clicked_news不够num_clicked_news_per_user条,补0(待考虑)
        pad_vec = torch.zeros(config.num_words_per_news, dtype=torch.long)
        clicked_news.extend([{'title': pad_vec, 'entities': pad_vec, 'category': torch.tensor(0)}
                             for _ in range(config.num_clicked_news_per_user - len(clicked_news))])
        for impression in impressions:
            if self.mode == 'test':
                candidate_id = impression
                is_click = torch.tensor(0.5)
            else:
                candidate_id, is_click = impression.split('-')
                is_click = torch.tensor(int(is_click)).float()
            candidate = self.news.loc[candidate_id]
            split_title = split_words(candidate.Title)
            candidate_category = torch.tensor(self.category_dict.get(candidate.Category, 0)).long()
            candidate_title = self.title_dict.wors2token(split_title)
            entities = candidate.Title_Entities
            entities = [] if pd.isna(entities) else json.loads(entities)
            entities = get_entities_from_title(split_title, entities)
            candidate_title_entities = self.entity_dict.wors2token(entities)
            to_return = {
                'candidate_news': {
                    'title': candidate_title,
                    'entities': candidate_title_entities,
                    'is_click': is_click,
                    'category': candidate_category
                },
                'clicked_news': clicked_news
            }
            yield to_return
