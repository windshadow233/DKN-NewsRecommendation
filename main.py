from torch.utils.data import DataLoader
from data.dataset import WordsTokenConverter, UserDataset, user_data_collate
from model.DKN import *
from config import config

model = DKN(config, "data/title_entities_embedding.pkl")
converter1 = WordsTokenConverter()
converter2 = WordsTokenConverter("data/title_entities_vocab.json")
dataset = UserDataset(converter1, converter2)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=user_data_collate)
