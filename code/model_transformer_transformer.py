import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import GPT2LMHeadModel, GPT2Config

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
    from transformers import GPT2LMHeadModel, GPT2Config

class GPT2Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, gpt_model_name='gpt2'):
        super(GPT2Decoder, self).__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name, pad_token_id=50256)
        # ViT의 임베딩 크기와 GPT2의 임베딩 크기를 맞추는 레이어
        self.linear = nn.Linear(embed_size, self.gpt.config.n_embd)
        self.vocab_size = vocab_size

    def forward(self, features, captions=None, lengths=None):
        features = self.linear(features)
        # 이미지 특징을 GPT2의 첫 토큰으로 사용
        input_ids = torch.cat([features.unsqueeze(1), captions[:, :-1]], dim=1)
        outputs = self.gpt(input_ids, labels=captions)
        return outputs.logits

    def sample(self, features, max_length=20):
        sampled_indexes = []
        inputs = features.unsqueeze(1)  # 첫 번째 입력은 이미지 특징
    
        for i in range(max_length):
            # Transformer 모델을 사용하여 출력 예측
            outputs = self.forward(inputs)  # outputs: (batch_size, sequence_length, vocab_size)
            _, predicted = outputs[:, -1, :].max(dim=1)  # 마지막 단어의 가장 높은 확률을 가진 단어 선택
    
            # 예측된 단어를 시퀀스에 추가
            sampled_indexes.append(predicted.unsqueeze(1))
            inputs = torch.cat((inputs, predicted.unsqueeze(1)), dim=1)  # 다음 입력을 위해 현재 단어 추가
    
        sampled_indexes = torch.cat(sampled_indexes, dim=1)  # sampled_indexes: (batch_size, max_seq_length)
        return sampled_indexes