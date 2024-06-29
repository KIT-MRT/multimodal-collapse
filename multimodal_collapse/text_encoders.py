import torch

from torch import nn
from x_clip.tokenizer import SimpleTokenizer


class BytePairTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = SimpleTokenizer()

    def forward(self, texts, device="cuda"):
        with torch.no_grad():
            token_indices = self.tokenizer.tokenize(list(texts))
            return token_indices.to(device)

    def decode_tokens(self, tokens):
        with torch.no_grad():
            return self.tokenizer.decode(tokens)