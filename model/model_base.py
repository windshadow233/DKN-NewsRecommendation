import torch
from torch import nn
from torch.nn import functional as F
import pickle


class Embedding(nn.Module):
    def __init__(self, config, entity_embedding):
        super(Embedding, self).__init__()
        self.use_context = config.use_context
        self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
        with open(entity_embedding, 'rb') as f:
            entity_embedding = pickle.loads(f.read())
        self.entity_embedding = torch.tensor(entity_embedding).float()

        entity_embedding_dim = self.entity_embedding.shape[1]
        self.transform = nn.Sequential(
            nn.Linear(entity_embedding_dim, config.embedding_dim),
            nn.Tanh()
        )

    def forward(self, titles, entities):
        """
        :param titles: (batch_size, num_words_per_news)
        :param entities: (batch_size, num_words_per_news)
        :return: (batch_size, 2 or 3, num_words_per_news, embedding_dim)
        """
        if self.use_context:
            raise RuntimeError('context not given')
        titles_embedding = self.word_embedding(titles)
        entities_embedding = self.transform(F.embedding(entities, weight=self.entity_embedding))
        return torch.stack([titles_embedding, entities_embedding], dim=1)


class KCNN(nn.Module):
    def __init__(self, config, entity_embedding):
        super(KCNN, self).__init__()
        self.embedding = Embedding(config, entity_embedding)
        self.conv_filters = nn.ModuleDict({
            str(i): nn.Conv2d(3 if config.use_context else 2, config.num_filters,
                              (x, config.embedding_dim))
            for i, x in enumerate(config.window_sizes)
        })

    def forward(self, news):

        """
        Args:
          news:
            {
                "titles": batch_size * num_words_per_news,
                "entities":batch_size * num_words_per_news
            }

        Return: (batch_size, len(window_sizes) * num_filter)
        """

        titles = news.get('titles')
        entities = news.get('entities')
        multi_channel_embedding = self.embedding(titles, entities)
        pooled_vecs = []
        for conv in self.conv_filters.values():
            conved_result = conv(multi_channel_embedding).squeeze(-1)
            activated = F.relu(conved_result)
            pooled = activated.max(-1).values
            pooled_vecs.append(pooled)
        return torch.cat(pooled_vecs, dim=1)


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.weight = nn.Sequential(
            nn.Linear(2 * len(config.window_sizes) * config.num_filters, 16),
            nn.Linear(16, 1)
        )

    def forward(self, candidate_news_vec, clicked_news_vec):

        """
        Args:
          candidate_news_vec: (batch_size, len(window_sizes) * num_filter)
          clicked_news_vec: (num_clicked_news_per_user, batch_size, len(window_sizes) * num_filters)

        Return: user_embedding: (batch_size, len(window_sizes) * num_filter)
        """

        candidate_expanded = candidate_news_vec.expand(clicked_news_vec.shape[0], -1, -1)
        clicked_news_weights = F.softmax(
            self.weight(torch.cat([clicked_news_vec, candidate_expanded], dim=-1)).squeeze(-1).transpose(0, 1), dim=-1)
        # (batch_size, num_clicked_news_per_user)
        user_embedding = (clicked_news_weights.unsqueeze(1) @ clicked_news_vec.transpose(0, 1)).squeeze(1)
        return user_embedding
