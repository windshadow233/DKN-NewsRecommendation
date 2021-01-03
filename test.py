import torch
import tqdm
import os
import pandas as pd
from model.DKN import DKN
from config import Config
from data.dataset import TestDataset, user_data_collate


model_path = 'trained_model/model2'
config = Config(os.path.join(model_path, 'config.json'))
model = DKN(config, 'data/entities_embedding.pkl')
model.load_state_dict(torch.load(os.path.join(model_path, 'state_dict', 'model.pkl')))
model.eval()
test_dataset = TestDataset('data/title_words_vocab.json', 'data/entities_vocab.json', mode='test')
with torch.no_grad():
    with open('prediction.txt', 'w') as f:
        try:
            for data in tqdm.tqdm(test_dataset):
                impression_ID = next(data)
                candidate, history, _ = user_data_collate(data)
                pred = model(candidate, history).tolist()
                rank = pd.Series(pred).rank(ascending=False).astype(int).to_list()
                f.write(f'{impression_ID} {str(rank)}\n')
        except IndexError:
            pass
