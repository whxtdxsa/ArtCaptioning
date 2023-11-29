import torch
import torch.utils.data as data
from PIL import Image
import nltk
import os

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    

class CocoDataset(data.Dataset):
    def __init__(self, root, captions, vocab, transform=None):
        self.root = root  # 이미지가 위치한 디렉토리
        self.vocab = vocab  # 단어장 객체
        self.transform = transform  # 이미지 변환 (옵션)

        # 캡션 데이터 로드 및 처리
        with open(captions, "r") as f:
            lines = f.readlines()
            self.captions = []
            for line in lines:
                index = line.find(",")
                path = line[:index]
                caption = line[index + 1:]
                self.captions.append((path.strip(), caption.strip()))

    def __getitem__(self, index):
        path, caption = self.captions[index]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(caption.lower())
        numerical_caption = [self.vocab('<start>')]
        numerical_caption.extend([self.vocab(token) for token in tokens])
        numerical_caption.append(self.vocab('<end>'))
        target = torch.tensor(numerical_caption, dtype=torch.long)

        return image, target

    def __len__(self):
        return len(self.captions)