import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torch.nn.utils.rnn import pack_padded_sequence

class ViTEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ViTEncoder, self).__init__()
        # ViT 모델 로드
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # 임베딩 차원 조정을 위한 추가 레이어
        self.linear = nn.Linear(self.vit.config.hidden_size, embed_size)
        self.embed_size = embed_size

    def forward(self, images):
        # ViT는 이미지를 입력으로 받아 특징을 반환함
        features = self.vit(pixel_values=images).last_hidden_state
        # 특징 벡터의 평균을 계산하여 고정 크기 벡터로 변환
        features = features.mean(dim=1)
        features = self.linear(features)
        return features
    
class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
