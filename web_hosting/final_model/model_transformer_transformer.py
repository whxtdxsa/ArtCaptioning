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

class GPT2Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, gpt2_model_name='gpt2'):
        super(GPT2Decoder, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

        # ViT의 출력을 GPT-2의 입력 크기에 맞게 조정하는 레이어
        self.linear = nn.Linear(embed_size, self.gpt2.config.n_embd)
        self.embed_size = embed_size

    def forward(self, features, captions, lengths):
        # 이미지 특징을 GPT-2 토큰 크기에 맞게 변환
        transformed_features = self.linear(features).unsqueeze(1)  # (batch_size, 1, hidden_size)

        # 캡션의 임베딩을 가져옴
        embeddings = self.gpt2.transformer.wte(captions)  # (batch_size, max_length, hidden_size)

        # 이미지 특징과 캡션 임베딩을 결합
        inputs_embeds = torch.cat([transformed_features, embeddings], dim=1)  # (batch_size, max_length+1, hidden_size)

        # 모델 출력 크기를 타겟 크기에 맞게 조정
        outputs = self.gpt2(inputs_embeds=inputs_embeds).logits
        packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True, enforce_sorted=False).data
        return packed_outputs 


    def sample(self, features, max_length=20):
        # 이미지 특징을 GPT-2 토큰 크기에 맞게 변환
        transformed_features = self.linear(features).unsqueeze(1)  # (batch_size, 1, hidden_size)

        generated = transformed_features
        result_ids = []

        for i in range(max_length):
            outputs = self.gpt2(inputs_embeds=generated)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            result_ids.append(next_token)
            next_token_embedding = self.gpt2.transformer.wte(next_token)
            generated = torch.cat([generated, next_token_embedding], dim=1)

        return torch.cat(result_ids, dim=1)
