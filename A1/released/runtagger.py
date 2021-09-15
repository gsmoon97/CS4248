# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import json
import numpy as np


def read_model_file(model_file):
    jdata = ''
    with open(model_file, 'r') as file:
        jdata = file.read()
    matrices = json.loads(jdata)
    return matrices


def read_test_file(test_file):
    test_data = []
    with open(test_file, 'r') as file:
        test_data = [line.split() for line in file]
    return test_data


def run_tagger(test_sentence, transition_matrix, emission_matrix):
    tags = list(transition_matrix.keys())
    tags.remove('<s>')
    back_pointers = np.zeros((len(test_sentence), len(tags)))
    state_probs = np.zeros((len(test_sentence), len(tags)))
    # initialize probabilities for the first word
    for (i, tag) in enumerate(tags):
        first_word = test_sentence[0]
        transition_prob = transition_matrix['<s>'][tag]
        if first_word not in emission_matrix[tag]:
            emission_prob = 0
        else:
            emission_prob = emission_matrix[tag][first_word]
        state_probs[0][i] = transition_prob * emission_prob
    # run vietrbi algorithm on the remaining words
    for i in range(1, len(test_sentence)):
        for (j, tag) in enumerate(tags):
            word = test_sentence[i]
            # print('word : {}'.format(word))
            # compare probabilities for different previous tags
            combined_probs = np.array([prev_tag_prob * transition_matrix[tags[prev_tag_idx]][tag]
                                       for prev_tag_idx, prev_tag_prob in enumerate(state_probs[i - 1])])
            # print('combined probs : {}'.format(combined_probs))
            max_idx = np.argmax(combined_probs)
            max_prob = max(combined_probs)
            back_pointers[i][j] = max_idx
            # compute emission probability
            if word not in emission_matrix[tag]:
                emission_prob = 0
            else:
                emission_prob = emission_matrix[tag][word]
    print(state_probs)
    print(np.argmax(state_probs[-1]))
    return


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    matrices = read_model_file(model_file)
    transition_matrix = matrices[0]
    emission_matrix = matrices[1]
    test_data = read_test_file(test_file)
    tagged_test_sentence = run_tagger(
        test_data[0], transition_matrix, emission_matrix)
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
