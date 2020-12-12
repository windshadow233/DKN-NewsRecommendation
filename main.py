import torch
from torch.nn.utils.rnn import pad_sequence
from data.dataset import WordsTokenConverter, UserDataset
from model.model_base import *
from config import config

model = KCNN(config, 'data/title_entities_embedding.pkl')
converter1 = WordsTokenConverter()
converter2 = WordsTokenConverter('data/title_entities_vocab.json')
data = UserDataset(converter1, converter2)
titles = data[0].get('titles')
entities = data[0].get('entities')
model({'titles': torch.tensor(titles), 'entities': torch.tensor(entities)})
