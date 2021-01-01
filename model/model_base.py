import torch
from torch import nn
from torch.nn import functional as F
import pickle
from transformers import AutoTokenizer, AutoModel


class Embedding(nn.Module):
    def __init__(self, config, entity_embedding):
        super(Embedding, self).__init__()
        self.use_context = config.use_context
        self.word_embedding = nn.Embedding(config.vocab_size, config.title_words_embedding_dim,
                                           padding_idx=config.pad_idx)
        # self.word_embedding = AutoModel.from_pretrained('./bert-base-uncased').embeddings.word_embeddings.eval()
        with open(entity_embedding, 'rb') as f:
            entity_embedding = pickle.loads(f.read())
        self.register_buffer('entity_embedding', torch.tensor(entity_embedding).float())

        entity_embedding_dim = self.entity_embedding.shape[1]
        self.transform = nn.Sequential(
            nn.Linear(entity_embedding_dim, config.title_words_embedding_dim),
            nn.Tanh()
        )

    def forward(self, titles, entities):
        """
        :param titles: (batch_size, num_words_per_news)
        :param entities: (batch_size, num_words_per_news)
        :return: (batch_size, 2 or 3, num_words_per_news, title_words_embedding_dim)
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
                              (x, config.title_words_embedding_dim))
            for i, x in enumerate(config.window_sizes)
        })
        if config.use_category:
            self.category_dense = nn.Sequential(
                nn.Linear(config.category_num, config.category_vec_dim),
                nn.ReLU(inplace=True)
            )
            self.category_num = config.category_num
            self.subcat_dense = nn.Sequential(
                nn.Linear(config.subcategory_num, config.subcategory_vec_dim),
                nn.ReLU(inplace=True)
            )
            self.subcat_num = config.subcategory_num

    def forward(self, news):

        """
        Args:
          news:
            {
                "titles": batch_size * num_words_per_news,
                "entities": batch_size * num_words_per_news,
                "categories": batch_size,
                "subcategories": batch_size
            }

        Return: (batch_size, len(window_sizes) * num_filters (|+category_vec_dim))
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
        out_vec = torch.cat(pooled_vecs, dim=1)
        if hasattr(self, 'category_dense'):
            categories = news.get('categories')
            categories_vec = torch.eye(n=self.category_num, device=categories.device)[categories]
            categories_vec = self.category_dense(categories_vec)
            subcats = news.get('subcategories')
            subcats_vec = torch.eye(n=self.subcat_num, device=subcats.device)[subcats]
            subcats_vec = self.subcat_dense(subcats_vec)
            out_vec = torch.cat([out_vec, categories_vec, subcats_vec], dim=1)
        return out_vec


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        in_features = 2 * len(config.window_sizes) * config.num_filters
        if config.use_category:
            in_features += 2 * config.category_vec_dim + 2 * config.subcategory_vec_dim
        self.weight = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, candidate_news_vec, clicked_news_vec):

        """
        Args:
          candidate_news_vec: (batch_size, len(window_sizes) * num_filters (|+category_vec_dim))
          clicked_news_vec: (num_clicked_news_per_user, batch_size, len(window_sizes) * num_filters (|+category_vec_dim))

        Return: user_embedding: (batch_size, len(window_sizes) * num_filters)
        """

        candidate_expanded = candidate_news_vec.expand(clicked_news_vec.shape[0], -1, -1)
        clicked_news_weights = F.softmax(
            self.weight(torch.cat([clicked_news_vec, candidate_expanded], dim=-1)).squeeze(-1).transpose(0, 1), dim=-1)
        # (batch_size, num_clicked_news_per_user)
        user_embedding = (clicked_news_weights.unsqueeze(1) @ clicked_news_vec.transpose(0, 1)).squeeze(1)
        return user_embedding
