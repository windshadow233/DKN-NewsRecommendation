from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from data.dataset import WordsTokenConverter, UserDataset, user_data_collate
from model.DKN import *
from config import config

device = torch.device(config.device)
model = DKN(config, "data/title_entities_embedding.pkl").to(device)
converter1 = WordsTokenConverter()
converter2 = WordsTokenConverter("data/title_entities_vocab.json")
dataset = UserDataset(converter1, converter2)
loss_fcn = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.99))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=user_data_collate)
for candidate_news, clicked_news, is_click in tqdm.tqdm(dataloader, total=len(dataloader)):
    optimizer.zero_grad()
    is_click = is_click.to(device)
    pred = model(candidate_news, clicked_news, False)
    loss = loss_fcn(pred, is_click)
    print(loss.item())
    loss.backward()
    optimizer.step()

