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
            self.label_class = list(set(self.labels))
            self.label_class_to_idx = {label_class: i for (i, label_class)
                                       in enumerate(self.label_class)}
        else:
            if vocab:
                self.label_class = vocab[1]
        self.vocab = None
        if vocab:
            self.vocab = vocab[0]
        else:
            char_dict = {}
            for text in self.texts:
                for i in range(len(text)):
                    if text[i].isalpha():
                        # count occurrence of each character (count for only letters)
                        char_dict[text[i]] = char_dict.get(text[i], 0) + 1
            # only include characters with 5+ occurences to the character list
            char_list = [char for char, count in char_dict.items()
                         if count >= 5]
            # build vocabulary of character bigrams using character list
            self.vocab = [x + y for x in char_list for y in char_list]
        # start index from 1 (index 0 reserved for <UNK>)
        self.bigram_to_idx = {bigram: i + 1 for (i, bigram)
                              in enumerate(self.vocab)}

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(self.vocab)
        num_class = len(self.label_class)
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
        for idx in range(len(raw_text) - 1):
            first = raw_text[idx]
            second = raw_text[idx+1]
            if first.isalpha() and second.isalpha():
                # consider only letters for the bigram (exclude punctuations, digits, etc.)
                # if <UNK>, index is 0
                raw_bigrams.append(self.bigram_to_idx.get(first + second, 0))
        text = torch.LongTensor(raw_bigrams)
        # find the index of the corresponding label
        if self.labels:
            raw_label = self.labels[i]
            label = torch.LongTensor([self.label_class_to_idx[raw_label]])
        else:
            label = torch.LongTensor([-1])
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
        dimension_d = 16  # arbitrary
        dimension_m = 16  # arbitrary
        # index 0 gets embedded to zero vector
        self.embeddings = nn.Embedding(num_vocab, dimension_d, padding_idx=0)
        # first feed-forward layer (hidden)
        self.linear1 = nn.Linear(dimension_d, dimension_m)
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # second feed-forward layer (final)
        self.linear2 = nn.Linear(dimension_m, num_class)

    def forward(self, x):
        # define the forward function here
        # obtain the corresponding embedding vectors for the given bigram from the lookup table (embedding matrix)
        # take as input, a mini-batch of examples represented by bigram indices
        emb_vectors = self.embeddings(x)
        # average bigram embeddings to obtain a single vector for each text (consider only non-zero paddings)
        input_vector = (emb_vectors.sum(dim=1)
                        / emb_vectors.count_nonzero(dim=1))
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
    criterion = nn.CrossEntropyLoss()
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
            optimizer.zero_grad()
            # do forward propagation
            output = model(texts)
            # do loss calculation
            loss = criterion(output, labels)
            # do backward propagation
            loss.backward()
            # do parameter optimization step
            optimizer.step()
            # calculate running loss value for non padding
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (step + 1)
            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()

    # save the model weight in the checkpoint variable
    # and dump it to system on the model_path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'vocab': [dataset.vocab, dataset.label_class],
        'lang_map': dataset.label_class_to_idx
    }
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
            for output in outputs:
                softmax_output = F.softmax(output, dim=0)
                label_class_idx = torch.argmax(softmax_output).item()
                labels.append(class_map[label_class_idx])

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
        learning_rate = 0.1
        batch_size = 5
        num_epochs = 30

        train(model, dataset, batch_size, learning_rate,
              num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        # load checkpoint
        checkpoint = torch.load(args.model_path)
        # the lang map should map the class index to the language id (e.g. eng, fra, etc.)
        lang_map = {idx: lang for lang, idx
                    in checkpoint['lang_map'].items()}
        # create the test dataset object using LangDataset class
        vocab = checkpoint['vocab']
        dataset = LangDataset(args.text_path, vocab=vocab)
        # initialize and load the model
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

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
