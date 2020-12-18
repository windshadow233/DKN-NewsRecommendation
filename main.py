from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import random
from data.dataset import WordsTokenConverter, TrainDataset, user_data_collate, TestDataset
from model.DKN import *
from config import config

torch.manual_seed(10)
random.seed(10)
device = torch.device(config.device)
model = DKN(config, "data/entities_embedding.pkl").to(device)
converter1 = WordsTokenConverter('data/title_words_vocab.json')
converter2 = WordsTokenConverter("data/entities_vocab.json")
train_set = TrainDataset(converter1, converter2, positive_rate=0.5)
loss_fcn = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.98))
train_loader = DataLoader(train_set, batch_size=64, shuffle=False, collate_fn=user_data_collate)
epochs = 5
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (candidate_news, clicked_news, is_click) in tqdm.tqdm(enumerate(train_loader), total=total_step):
        optimizer.zero_grad()
        pred = model(candidate_news, clicked_news, sigmoid_at_end=False)
        loss = loss_fcn(pred, is_click)
        loss.backward()
        optimizer.step()
        print(loss.item())
