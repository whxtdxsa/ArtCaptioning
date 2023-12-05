import os
import nltk
from PIL import Image
import torch
import torch.utils.data as data

class CocoDataset(data.Dataset):
    def __init__(self, root, augmented_root, captions, vocab, transform=None):
        self.root = root
        self.augmented_root = augmented_root
        self.vocab = vocab
        self.transform = transform

        with open(captions, "r") as f:
            lines = f.readlines()
            self.captions = []
            for line in lines:
                index = line.find(",")
                path = line[:index]
                caption = line[index + 1:]
                # 원본 이미지 캡션 추가
                self.captions.append((os.path.join(root, path.strip()), caption.strip()))
                if augmented_root != "none":
                    # 블러 처리된 이미지 캡션 추가
                    self.captions.append((os.path.join(augmented_root, path.strip()), caption.strip()))

    def __getitem__(self, index):
        image_path, caption = self.captions[index]
        image = Image.open(image_path).convert('RGB')

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

# Define a function to create batches from a list of tuples (image, caption)
def collate_fn(data):
    # Sort data by caption length in descending order for efficient packing and padding
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images from tuple of 3D tensors to a 4D tensor (batch size, color channels, height, width)
    images = torch.stack(images, 0)

    # Create a tensor to hold the padded captions with maximum length in the batch
    lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    # Fill the tensor with caption tokens; padding will be automatically added where necessary
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths  # Return the batched images, padded captions, and lengths of each caption

# Define a separate collate function for testing where we don't want to sort the captions
def collate_fn_test(data):
    # Keep the order of data as is (useful during testing)
    images, captions = zip(*data)

    # Merge images into a 4D tensor as before
    images = torch.stack(images, 0)

    # Create a tensor for the padded captions as before
    lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    # Pad the captions with the original tokens up to the length of each caption
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths  # Return images, padded captions, and lengths as before

def get_loader(root, augmented_root, captions, vocab, transform, batch_size, shuffle, num_workers, testing, pin_memory=False):
    coco_dataset = CocoDataset(root=root, augmented_root=augmented_root, captions=captions, vocab=vocab, transform=transform)

    # 테스트 중인지에 따라 적절한 collate 함수 선택
    collate_fn_to_use = collate_fn_test if testing else collate_fn

    # DataLoader를 반환합니다.
    data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_to_use, 
                                              pin_memory=pin_memory)
    return data_loader