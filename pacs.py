# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:48:56 2020

@author: marco
"""

from torchvision.datasets import ImageFolder

class pacs():
  def __init__(self, root, domain='photo', transform=None, target_transform=None):
        super(pacs, self).__init__(root, transform=transform, target_transform=target_transform)

        self.domain = domain
        self.labels = []
        self.images = []

        for data in ImageFolder(root=root + '/' + domain):
            self.images.append(data[0])
            self.labels.append(data[1])
        
  def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        image = self.images[index]
        label = self.labels[index]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return image, label

  def __len__(self):
      '''
      The __len__ method returns the length of the dataset
      It is mandatory, as this is used by several other components
      '''
      #length = ... # Provide a way to get the length (number of elements) of the dataset
      return len(self.images)
          