from torch import nn
from transformers import ViTModel
from torchvision.transforms.functional import resize


class PretrainedViT(nn.Module):
    def __init__(self, weights="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.model = ViTModel.from_pretrained(weights)

    def forward(self, x):
        x = resize(x, size=(224, 224))
        return self.model(x).last_hidden_state
