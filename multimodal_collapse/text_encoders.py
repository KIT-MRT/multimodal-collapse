import torch

from torch import nn
from x_clip.tokenizer import SimpleTokenizer


class BytePairTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = SimpleTokenizer()

    def forward(self, texts):
        with torch.no_grad():
            return self.tokenizer.tokenize(list(texts))

    def decode_tokens(self, tokens):
        with torch.no_grad():
            return self.tokenizer.decode(tokens)