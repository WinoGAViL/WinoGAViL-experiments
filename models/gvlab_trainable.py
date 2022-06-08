import torch
from torch import nn

class BaselineModel(nn.Module):

    def __init__(self, backend_model):
        super(BaselineModel, self).__init__()
        self.backend_model = backend_model
        pair_embed_dim = self.backend_model.text_dim + self.backend_model.image_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_embed_dim),
            nn.Linear(pair_embed_dim, int(pair_embed_dim/4)), # for ViT-B/32 it will be 256
            nn.ReLU(),
            nn.Linear(int(pair_embed_dim/4), 1)
        )

    def forward(self, input_image_vector, text_vector):
        concatenated = []
        for i in range(len(input_image_vector)):
            input_concat = torch.cat([input_image_vector[i], text_vector[i]], dim=1)
            concatenated.append(input_concat)

        x = torch.cat(concatenated).to(torch.float32)
        x = self.classifier(x)
        return x


