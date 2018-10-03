import torch.nn as nn

class PhonemeEmbedding(nn.Module):
    def __init__(self, n_phoneme, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim * 2 * 3
        self.hidden_size = embedding_dim
        self.embedding = nn.Embedding(n_phoneme, self.embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=3),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        embed_x = self.embedding(x)
        embed_x = embed_x.view(-1, 1, 6, self.embedding_dim) # (B * C) * (3 * 2) * (H * 3 * 2)
        outputs = self.conv(embed_x)
        return outputs.view(batch_size, -1, self.hidden_size) # B * C * H



