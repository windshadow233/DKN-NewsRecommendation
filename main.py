from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import sys
import random
from data.dataset import TrainDataset, user_data_collate
from model.DKN import *
from config import config

torch.manual_seed(10)
random.seed(10)
device = torch.device(config.device)
model = DKN(config, "data/entities_embedding.pkl").to(device)
train_set = TrainDataset('data/title_words_vocab.json', 'data/entities_vocab.json', positive_rate=0.167)
loss_fcn = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.98))
train_loader = DataLoader(train_set, batch_size=64, shuffle=False, collate_fn=user_data_collate)
epochs = 10
total_step = len(train_loader)
for epoch in range(epochs):
    for candidate_news, clicked_news, is_click in tqdm.tqdm(train_loader,
                                                            total=total_step,
                                                            desc='Epoch_%s' % epoch):
        optimizer.zero_grad()
        pred = model(candidate_news, clicked_news, sigmoid_at_end=False)
        loss = loss_fcn(pred, is_click)
        loss.backward()
        optimizer.step()
        print('\r', loss.item(), flush=True, file=sys.stdout)
