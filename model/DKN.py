from .model_base import *


class DKN(nn.Module):
    def __init__(self, config, entity_embedding):
        super(DKN, self).__init__()
        self.kcnn = KCNN(config, entity_embedding=entity_embedding)
        self.attention = Attention(config)
        in_features = 2 * len(config.window_sizes) * config.num_filters
        if config.use_category:
            in_features += 2 * config.category_vec_dim + 2 * config.subcategory_vec_dim
        self.click_prob = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, candidate_news, clicked_news, sigmoid_at_end=True):

        """
        Args:
          candidate_news:
            {
                "titles": batch_size * num_words_per_news,
                "entities": batch_size * num_words_per_news,
                "categories": batch_size,
                "subcategories": batch_size
            }
          clicked_news:
            [
                {
                    "titles": batch_size * num_words_per_news,
                    "entities": batch_size * num_words_per_news,
                    "categories": batch_size * num_words_per_news,
                    "subcategories": batch_size
                } * num_clicked_news_per_user
            ]
         sigmoid_at_end: use sigmoid at last

        Returns:
          click_prob: batch_size

        """

        candidate_news_vec = self.kcnn(candidate_news)  # (batch_size, len(window_sizes) * num_filter (|+category_vec_dim + subcategory_vec_dim))
        clicked_news_vec = torch.stack([
            self.kcnn(news) for news in clicked_news
        ])
        user_embedding = self.attention(candidate_news_vec, clicked_news_vec)
        # (batch_size, len(window_sizes) * num_filter (|+category_vec_dim))
        click_prob = self.click_prob(torch.cat([user_embedding, candidate_news_vec], dim=1)).squeeze(dim=1)
        if sigmoid_at_end:
            click_prob = click_prob.sigmoid()
        return click_prob
