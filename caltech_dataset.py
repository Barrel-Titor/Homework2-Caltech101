from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        if split != 'train' and split != 'test':
            raise ValueError("should take 'train' or 'test' as value of 'split'")
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.img_list = []

        self.label_list = [x.name for x in os.scandir(self.root)]
        self.label_list.sort()
        self.label_list.remove('BACKGROUND_Google')

        self.split_ = os.path.join(self.root, '..', self.split + '.txt')

        with open(self.split_, 'r') as f:
            for line in f:
                line = line.strip('\n')
                label = line.split('/')[0]
                if label != 'BACKGROUND_Google':
                    self.img_list.append((line, self.label_list.index(label)))

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int
        path, label = self.img_list[index]
        image = pil_loader(os.path.join(self.root, path))

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.img_list) # Provide a way to get the length (number of elements) of the dataset
        return length
