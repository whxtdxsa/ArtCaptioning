import torch
from torchvision import transforms
from PIL import Image
import vocabulary 
import pickle
import final_model.resnetLstm as model
import os

# 모델 파라미터 설정
embed_size = 256
hidden_size = 1024
num_layers = 1
vocab_path = "./final_model/vocab.pkl"  # Path to the preprocessed vocabulary file

# Load vocabulary file
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# 파일 관리를 위한 함수
def manage_files(directory, max_files):
    files = os.listdir(directory)
    while len(files) > max_files:
        full_paths = [os.path.join(directory, file) for file in files]
        oldest_file = min(full_paths, key=os.path.getctime)
        os.remove(oldest_file)


def load_model(encoder_path, decoder_path, embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(vocab), num_layers=num_layers):
    # 모델 초기화 및 가중치 로드
    encoder = model.EncoderCNN(embed_size)
    decoder = model.DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def generate_caption(image_path, encoder, decoder, vocab=vocab, max_length=20):
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # 디바이스 설정 (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 특징 추출 및 캡션 생성
    with torch.no_grad():
        features = encoder(image)
        sampled_ids = decoder.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()

    # ID를 단어로 변환
    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        caption.append(word)
        if word == '<end>':
            break

    return ' '.join(caption[1:-1]).capitalize()

