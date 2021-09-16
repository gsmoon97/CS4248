# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
from collections import defaultdict
import json


def read_train_file(train_file):
    train_data = []
    with open(train_file, 'r') as file:
        train_data = [line.split() for line in file]
    train_data = [[tagged_word.rsplit(
        '/', 1) for tagged_word in tagged_sentence] for tagged_sentence in train_data]
    return train_data


def build_tag_set(train_data):
    tag_set = set()
    for tagged_sentence in train_data:
        for tagged_word in tagged_sentence:
            tag_set.add(tagged_word[1])
    return tag_set


def build_voc_set(train_data):
    voc_set = set()
    for tagged_sentence in train_data:
        for tagged_word in tagged_sentence:
            voc_set.add(tagged_word[0])
    return voc_set


def build_transition_matrix(train_data, tag_set):
    transition_matrix = {
        pre_tag: {cur_tag: 0 for cur_tag in tag_set} for pre_tag in tag_set
    }
    # add <s> as one of the previous tags (states) of the transition matrix
    transition_matrix['<s>'] = {cur_tag: 0 for cur_tag in tag_set}
    for tagged_sentence in train_data:
        for i in range(len(tagged_sentence)):
            if i == 0:
                transition_matrix['<s>'][tagged_sentence[i][1]] += 1
            else:
                transition_matrix[tagged_sentence[i-1]
                                  [1]][tagged_sentence[i][1]] += 1
    for pre_tag, cur_tags in transition_matrix.items():
        total_count_for_pre_tag = sum(cur_tags.values())
        for cur_tag, count in transition_matrix[pre_tag].items():
            transition_matrix[pre_tag][cur_tag] = count / \
                total_count_for_pre_tag
    return transition_matrix


def build_emission_matrix(train_data, tag_set, voc_set):
    emission_matrix = {
        cur_tag: {cur_voc: 0 for cur_voc in voc_set} for cur_tag in tag_set
    }
    for tagged_sentence in train_data:
        for i in range(len(tagged_sentence)):
            emission_matrix[tagged_sentence[i][1]][tagged_sentence[i][0]] += 1
    # add-one smoothing for emission probabilities
    # for tag_idx in emission_matrix:
    #     for word_idx in emission_matrix[tag_idx]:
    #         emission_matrix[tag_idx][word_idx] += 1
    for cur_tag, cur_words in emission_matrix.items():
        total_count_for_cur_tag = sum(cur_words.values())
        for cur_word, count in emission_matrix[cur_tag].items():
            emission_matrix[cur_tag][cur_word] = count / \
                total_count_for_cur_tag
    return emission_matrix


def write_model_file(model_file, input):
    with open(model_file, 'w') as file:
        file.write(json.dumps(input))


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    train_data = read_train_file(train_file)
    voc_set = build_voc_set(train_data)
    tag_set = build_tag_set(train_data)
    transition_matrix = build_transition_matrix(train_data, tag_set)
    emission_matrix = build_emission_matrix(train_data, tag_set, voc_set)
    matrices = [transition_matrix, emission_matrix]
    write_model_file(model_file, matrices)
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
