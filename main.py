import torch
from torch.utils.data import DataLoader
from data.dataset import WordsTokenConverter, UserDataset
from model.DKN import *
from config import config

model = DKN(config, "data/title_entities_embedding.pkl")
converter1 = WordsTokenConverter()
converter2 = WordsTokenConverter("data/title_entities_vocab.json")
dataset = UserDataset(converter1, converter2)
# model({"titles": torch.tensor(titles), "entities": torch.tensor(entities)})


def user_data_collate(one_batch):
    """
    Args:
        one_batch:
            ({
                "candidate_news": {
                    "title": num_words_per_news,
                    "entities": num_words_per_news,
                    "is_click": 1
                },
                "clicked_news": {
                    "titles": [num_words_per_news] * num_clicked_news_per_user,
                    "entities": [num_words_per_news] * num_clicked_news_per_user
                }
            } * batch_size)
    Return:
        candidate_news:
            {
                "titles": batch_size * num_words_per_news,
                "entities": batch_size * num_words_per_news
            }
        clicked_news:
            [
                {
                    "titles": batch_size * num_words_per_news,
                    "entities": batch_size * num_words_per_news
                } * num_clicked_news_per_user
            ]
        is_click: batch_size
    """
    clicked_news_titles = [[] for _ in range(config.num_clicked_news_per_user)]
    clicked_news_entities = [[] for _ in range(config.num_clicked_news_per_user)]
    candidate_news_titles = []
    candidate_news_entities = []
    is_click = []
    for data in one_batch:
        candidate = data['candidate_news']
        clicked = data['clicked_news']
        candidate_news_titles.append(candidate['title'])
        candidate_news_entities.append(candidate['entities'])
        is_click.append(candidate['is_click'])
        for i, (clicked_titles, clicked_entities) in enumerate(zip(clicked['titles'], clicked['entities'])):
            clicked_news_titles[i].append(clicked_titles)
            clicked_news_entities[i].append(clicked_entities)
    clicked_news_titles = list(map(lambda x: torch.stack(x), clicked_news_titles))
    clicked_news_entities = list(map(lambda x: torch.stack(x), clicked_news_entities))
    candidate_news = {
        'titles': torch.stack(candidate_news_titles),
        'entities': torch.stack(candidate_news_entities),
        'is_click': torch.stack(is_click)
    }
    clicked_news = [{'titles': titles, 'entities': entities}
                    for titles, entities in zip(clicked_news_titles, clicked_news_entities)]
    return candidate_news, clicked_news


dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=user_data_collate)
t,en=next(iter(dataloader))
