import torch
import tqdm
import numpy as np
import json
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from model.DKN import DKN
from config import Config
from data.dataset import TestDataset, user_data_collate


model_path = 'trained_model/model2'
config = Config(os.path.join(model_path, 'config.json'))
model = DKN(config, 'data/entities_embedding.pkl')
model.load_state_dict(torch.load(os.path.join(model_path, 'state_dict', 'model.pkl')))
model.eval()
dev_dataset = TestDataset('data/title_words_vocab.json', 'data/entities_vocab.json', mode='dev')
auc = 0
with torch.no_grad():
    for i, data in enumerate(tqdm.tqdm(dev_dataset), 1):
        try:
            next(data)
            candidate, history, truth = user_data_collate(data)
            truth = truth.cpu().numpy()
            pred = model(candidate, history).cpu().numpy()
            auc_score = roc_auc_score(truth, pred)
            auc += auc_score
            print(f'AUC: {auc_score} | Avg: {auc / i}')
        except IndexError:
            break
"""直接用点击量作为无history用户新闻的score"""
with open('data/news_clicked_freq.json', 'r') as f:
    news_freq = json.loads(f.read())
behaviors = pd.read_csv('data/dev/behaviors.tsv', sep='\t', header=None)
behaviors = behaviors[behaviors[3].isna()]
for impressions in tqdm.tqdm(behaviors[4]):
    impressions = impressions.split(' ')
    truth = np.array([int(impression.split('-')[1]) for impression in impressions])
    score = np.array([news_freq.get(impression.split('-')[0], 0) for impression in impressions])
    auc_score = roc_auc_score(truth, score)
    auc += auc_score
    print(auc_score)
print(f'Avg: {auc / (i + len(behaviors))}')
