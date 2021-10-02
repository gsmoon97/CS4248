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

torch.manual_seed(0)


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
        raw_label = self.labels[i]
        label = torch.LongTensor([self.label_to_idx[raw_label]])
        return text, label


class Model(nn.Module):
    """
    Define a model that with one embedding layer, a hidden
    feed-forward layer, a dropout layer, and a feed-forward
    layer that reduces the dimension to num_class
    """

    def __init__(self, num_vocab, num_class, dropout=0.3):
        super().__init__()
        # define your model here
        # embedding layer
        dimension_d = 100  # temporary
        dimension_m = 50  # temporary
        self.embeddings = nn.Embedding(num_vocab, dimension_d)
        # first feed-forward layer (hidden)
        self.linear1 = nn.Linear(dimension_d, dimension_m)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # second feed-forward layer (final)
        self.linear2 = nn.Linear(dimension_m, num_class)

    def forward(self, x):
        # define the forward function here
        # obtain the corresponding embedding vector for the given bigram from the lookup table (embedding matrix)
        emb_vectors = [self.embeddings(x_i) for x_i in x]
        # average bigram embeddings to obtain a single vector
        input_vector = emb_vectors.mean(dim=0)
        # feed the input vector to the first layer (hidden)
        # apply ReLU activation function to obtain the hidden vector
        hidden_vector = F.relu(self.linear1(input_vector))
        # apply dropout to the hidden vector (during training)
        dropped_hidden_vector = self.dropout(hidden_vector)
        # feed the dropped-out hidden vector to the second layer (final)
        output_vector = self.linear2(dropped_hidden_vector)
        return output_vector


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


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and the optimizer with the specified learning rate and specified number of epoch.
    """
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = None
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients

            # do forward propagation

            # do loss calculation

            # do backward propagation

            # do parameter optimization step

            # calculate running loss value for non padding

            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()

    # save the model weight in the checkpoint variable
    # and dump it to system on the model_path
    # tip: the checkpoint can contain more than just the model
    checkpoint = None
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format(
        (end - start).seconds / 60.0))


def test(model, dataset, class_map, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20,
                             collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()
            # get the label predictions

    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = LangDataset(args.text_path, args.label_path)
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)

        # you may change these hyper-parameters
        learning_rate = None
        batch_size = None
        num_epochs = None

        train(model, dataset, batch_size, learning_rate,
              num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        # the lang map should map the class index to the language id (e.g. eng, fra, etc.)
        lang_map = None

        # create the test dataset object using LangDataset class

        # initialize and load the model

        # run the prediction
        preds = test(model, dataset, lang_map, device)

        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(preds))
    print('\n==== A2 Part 2 Done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None,
                        help='path to the label file')
    parser.add_argument('--train', default=False,
                        action='store_true', help='train the model')
    parser.add_argument('--test', default=False,
                        action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True,
                        help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt',
                        help='path to the output file during testing')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
