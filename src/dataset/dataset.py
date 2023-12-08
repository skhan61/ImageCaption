import os

import nltk
import torch
from nltk.tokenize import word_tokenize
from PIL import Image
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

# nltk.download('punkt')


class Vocabulary:
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word('<pad>')  # Padding
        self.add_word('<start>')  # Start token
        self.add_word('<end>')  # End token
        self.add_word('<unk>')  # Unknown token

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)


class CocoCaptionsDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root_dir, ann_file, vocab, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption to word ids.
        tokens = word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, img_id

    def __len__(self):
        return len(self.ids)

    def get_all_captions_for_image(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = [self.coco.anns[ann_id]['caption'] for ann_id in ann_ids]
        return captions

    def get_image_path(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        return os.path.join(self.root_dir, path)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, img_id)."""
    # Sort a data list by caption length (descending order)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Compute the length of the longest caption in the batch
    max_length = max([len(cap) for cap in captions])

    # Pad all captions to the length of the longest caption
    captions_padded = torch.zeros(len(captions), max_length).long()
    lengths = []
    for i, cap in enumerate(captions):
        end = len(cap)
        captions_padded[i, :end] = cap
        lengths.append(end)

    # Convert img_ids to a tensor
    img_ids = torch.tensor(img_ids)

    return images, captions_padded, torch.tensor(lengths), img_ids
