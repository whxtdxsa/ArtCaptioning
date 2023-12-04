import os
import json
import nltk
import pickle
from PIL import Image
from collections import Counter
from multiprocessing import Pool
from vocabulary import Vocabulary

# 필요한 변수 정의
num_train_img = 3000
num_val_img = 300
num_test_img = 300
word_threshold = 6

nltk.download('punkt')

# 이미지 경로 정의
image_paths = "./images"
train_img_path = "./dataset/train/images"
val_img_path = "./dataset/val/images"
test_img_path = "./dataset/test/images"
img_dir = [train_img_path, val_img_path, test_img_path]
size = [256, 256]

# 캡션 파일 및 저장 경로 정의
art_caption_path = "./ArtCap.json"
train_caption_path = "./dataset/train/captions.txt"
val_caption_path = "./dataset/val/captions.txt"
test_caption_path = "./dataset/test/captions.txt"
vocab_path = "./dataset/vocab.pkl"

# 이미지 리사이징 함수
def resize_image(image_path_tuple):
    img_path, output_dir, size = image_path_tuple
    with Image.open(img_path) as img:
        resized_img = img.resize(size, Image.LANCZOS)
        resized_img.save(os.path.join(output_dir, os.path.basename(img_path)))

# 이미지 병렬 처리 함수
def process_images_in_parallel(image_paths, output_dir, size, num_processes=4):
    image_path_tuples = [(img_path, output_dir, size) for img_path in image_paths]
    with Pool(num_processes) as p:
        p.map(resize_image, image_path_tuples)

# 디렉토리 생성
for path in [train_img_path, val_img_path, test_img_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# 이미지 파일 리스트 생성 및 분할
images = sorted(os.listdir(image_paths))
train_images = images[:num_train_img]
val_images = images[num_train_img:num_train_img + num_val_img]
test_images = images[num_train_img + num_val_img:num_train_img + num_val_img + num_test_img]

# 이미지 경로 생성 및 이미지 리사이징
train_image_paths = [os.path.join(image_paths, image) for image in train_images]
val_image_paths = [os.path.join(image_paths, image) for image in val_images]
test_image_paths = [os.path.join(image_paths, image) for image in test_images]
process_images_in_parallel(train_image_paths, train_img_path, size)
process_images_in_parallel(val_image_paths, val_img_path, size)
process_images_in_parallel(test_image_paths, test_img_path, size)

# 캡션 처리 함수
def process_captions(json_file_path, img_dirs):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    captions_data = []
    counter = Counter()

    for file_name, captions in data.items():
        for caption in captions:
            caption = caption.strip().replace("\n", " ")
            line = f"{file_name},{caption}\n"
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            captions_data.append(line)

    train_captions, val_captions, test_captions = [], [], []
    for img_dir, dataset in zip(img_dirs, [train_captions, val_captions, test_captions]):
        file_names = set(os.listdir(img_dir))
        for line in captions_data:
            if line.split(',')[0] in file_names:
                dataset.append(line)

    return train_captions, val_captions, test_captions, counter

# 캡션 데이터 처리 및 저장
train_captions, val_captions, test_captions, counter = process_captions(art_caption_path, img_dir)

def save_captions(captions, file_path):
    with open(file_path, 'w') as f:
        f.writelines(captions)

save_captions(train_captions, train_caption_path)
save_captions(val_captions, val_caption_path)
save_captions(test_captions, test_caption_path)


# 기존 COCO 데이터셋으로 생성된 vocab.pkl 파일 로드
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# 단어 빈도 필터링 및 어휘 사전 생성
words = [word for word, cnt in counter.items() if cnt >= word_threshold]


for word in words:
    vocab.add_word(word)

with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
