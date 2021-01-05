import torch
import tqdm
import json
import numpy as np
import os
import pandas as pd
from model.DKN import DKN
from config import Config
from data.dataset import TestDataset, user_data_collate


model_path = 'trained_model/model3'
config = Config(os.path.join(model_path, 'config.json'))
model = DKN(config, 'data/entities_embedding.pkl')
model.load_state_dict(torch.load(os.path.join(model_path, 'state_dict', 'model.pkl')))
model.eval()
test_dataset = TestDataset('data/title_words_vocab.json', 'data/entities_vocab.json', mode='test')
pred_dict = {}
with torch.no_grad():
    for data in tqdm.tqdm(test_dataset):
        try:
            impression_ID = next(data)
            candidate, history, _ = user_data_collate(data)
            pred = model(candidate, history).tolist()
            rank = pd.Series(pred).rank(method='first', ascending=False).astype(int).to_list()
            pred_dict[impression_ID] = str(rank)
        except IndexError:
            break
"""直接用点击量作为无history用户新闻的score"""
with open('data/news_clicked_freq.json', 'r') as f:
    news_freq = json.loads(f.read())
behaviors = pd.read_csv('data/test/behaviors.tsv', sep='\t', header=None).set_index(0)
behaviors = behaviors[behaviors[3].isna()]
for i in tqdm.tqdm(behaviors.index):
    impressions = behaviors.loc[i][4].split(' ')
    score = np.array([news_freq.get(impression.split('-')[0], 0) for impression in impressions])
    rank = pd.Series(score).rank(method='first', ascending=False).astype(int).to_list()
    pred_dict[i] = str(rank) + '\n'
pred_list = [(key, value) for key, value in pred_dict.items()]
pred_list.sort(key=lambda x: x[0])
pred_list = [' '.join([str(item[0]), item[1]]) for item in pred_list]
with open('prediction.txt', 'w') as f:
    f.writelines(pred_list)
