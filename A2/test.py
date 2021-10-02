import os
import re
import sys
import string
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class LangDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """

    def __init__(self, text_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            text_path (string): Path to the text file.
            label_path (string, optional): Path to the label file.
            vocab (string, optional): You may use or not use this
        """
        self.texts = None
        with open(text_path, 'r') as file:
            self.texts = [text.strip() for text in file]
        self.labels = None
        if label_path:
            with open(label_path, 'r') as file:
                self.labels = [label.strip() for label in file]
        self.vocab = None
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = set()
            for text in self.texts:
                for i in range(len(text) - 1):
                    first = text[i]
                    second = text[i+1]
                    if first.isalpha() and second.isalpha():
                        # add only letters to the vocabulary (exclude punctuations, digits, etc.)
                        self.vocab.add(first + second)
        self.bigram_to_idx = {bigram: i for (i, bigram)
                              in enumerate(self.vocab)}
        self.label_to_idx = {label: i for (i, label)
                             in enumerate(set(self.labels))}

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(self.vocab)
        num_class = len(set(self.labels))
        return num_vocab, num_class

    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).
        both text and label are recommended to be in pytorch tensor type.

        DO NOT pad the tensor here, do it at the collator function.
        """
        # find the indices of the corresponding bigrams
        raw_text = self.texts[i]
        raw_bigrams = []
        for i in range(len(raw_text) - 1):
            first = raw_text[i]
            second = raw_text[i+1]
            if first.isalpha() and second.isalpha():
                # consider only letters for the bigram (exclude punctuations, digits, etc.)
                raw_bigrams.append(self.bigram_to_idx[first + second])
        text = torch.LongTensor(raw_bigrams)
        # find the index of the corresponding label
        # print(f'{i}th label')
        raw_label = self.labels[i]
        label = torch.LongTensor([self.label_to_idx[raw_label]])
        return text, label


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    raw_texts = [ex[0] for ex in batch]
    raw_labels = [ex[1] for ex in batch]
    texts = torch.nn.utils.rnn.pad_sequence(raw_texts, batch_first=True)
    labels = torch.LongTensor(raw_labels)
    return texts, labels


def main():
    text_path = './x_train.txt'
    label_path = './y_train.txt'
    dataset = LangDataset(text_path, label_path)
    # print(f'Text : {dataset.texts}')
    # print(f'Label : {dataset.labels}')
    # print(f'Vocab : {dataset.vocab}')
    # print(f'Bigram to Index : dataset.bigram_to_idx')
    # print(f'Label to Index : dataset.label_to_idx')
    # num_vocab, num_class = dataset.vocab_size()
    # print(f'Number of Vocab : {num_vocab}')
    # print(f'Number of Class : {num_class}')
    # print(f'Number of Instances : {dataset.__len__()}')
    # i = 11
    # text_i, label_i = dataset.__getitem__(i)
    # print(f'{i}th Text : {text_i}')
    # print(f'{i}th Label : {label_i}')
    batch_size = 10
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    for step, data in enumerate(data_loader, 0):
        text = data[0]
        label = data[1]
        print(f'text : {text}')
        print(f'label : {label}')
        if step > 5:
            break


if __name__ == "__main__":
    main()
