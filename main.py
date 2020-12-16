from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from data.dataset import WordsTokenConverter, UserDataset, user_data_collate
from model.DKN import *
from config import config

torch.manual_seed(10)
device = torch.device(config.device)
model = DKN(config, "data/entities_embedding.pkl").to(device)
converter1 = WordsTokenConverter('data/title_words_vocab.json')
converter2 = WordsTokenConverter("data/entities_vocab.json")
dataset = UserDataset(converter1, converter2)
loss_fcn = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.98))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=user_data_collate)
for candidate_news, clicked_news, is_click in tqdm.tqdm(dataloader, total=len(dataloader)):
    optimizer.zero_grad()
    pred = model(candidate_news, clicked_news, sigmoid_at_end=False)
    loss = loss_fcn(pred, is_click)
    print(loss.item())
    loss.backward()
    optimizer.step()
