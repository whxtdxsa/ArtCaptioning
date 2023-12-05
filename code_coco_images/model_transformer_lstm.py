import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module): # ViT Encoder
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
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
    
class Decoder(nn.Module): # LSTM Decoder
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()
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

    def sample(self, features, max_length=20):
        sampled_ids = []
        inputs = features.unsqueeze(1)  # 첫 번째 입력은 이미지 특징
        states = None

        for i in range(max_length):
            # 현재의 입력에 대해 LSTM 레이어 실행
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))  # 선형 레이어를 통해 다음 단어 예측

            _, predicted = outputs.max(1)  # 가장 높은 확률을 가진 단어 선택
            sampled_ids.append(predicted.unsqueeze(1))

            # 다음 입력을 위해 현재 예측된 단어 추가
            inputs = self.embed(predicted).unsqueeze(1)

        sampled_ids = torch.cat(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids