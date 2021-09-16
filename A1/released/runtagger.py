# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import json
import numpy as np
import re


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


# def build_unknown_matrix(emission_matrix, tags):
#     unknown_matrix = {}
#     for i, tag in enumerate(tags):
#         total_word_count = 0
#         unknown_word_count = 0
#         for count in emission_matrix[tag].values():
#             if count == 1:
#                 unknown_word_count += count
#             total_word_count += count
#         unknown_matrix.update({tag: unknown_word_count/total_word_count})
#     return unknown_matrix


def build_capital_matrix(emission_matrix, tags):
    capital_matrix = {}
    for i, tag in enumerate(tags):
        total_word_count = 0
        capital_word_count = 0
        for word, count in emission_matrix[tag].items():
            if word[0].isupper():
                capital_word_count += count
            total_word_count += count
        capital_matrix.update({tag: capital_word_count/total_word_count})
    return capital_matrix


def build_suffix_matrix(emission_matrix, tags, suffixes):
    suffixes.append('N/A')
    suffix_matrix = {
        tag: {suffix: 0 for suffix in suffixes} for tag in tags
    }
    suffixes.remove('N/A')
    for i, tag in enumerate(tags):
        for word, count in emission_matrix[tag].items():
            suffix_found = False
            for suffix in suffixes:
                if bool(re.match(r'[A-Za-z]+{}$'.format(suffix), word)):
                    suffix_matrix[tag][suffix] += count
                    suffix_found = True
                    break
            if not suffix_found:
                suffix_matrix[tag]['N/A'] += count
    for tag, suffixes in suffix_matrix.items():
        total_count_for_tag = sum(emission_matrix[tag].values())
        for suffix, count in suffixes.items():
            suffix_matrix[tag][suffix] = count / total_count_for_tag
    return suffix_matrix


def run_tagger_on_sentence(test_sentence, tags, transition_matrix, emission_matrix, captial_matrix, suffix_matrix, suffixes):
    back_pointers = np.zeros((len(test_sentence), len(tags)), dtype=int)
    state_probs = np.zeros((len(test_sentence), len(tags)))
    # initialize probabilities for the first word
    for (i, tag) in enumerate(tags):
        first_word = test_sentence[0]
        transition_prob = transition_matrix['<s>'][tag]
        if first_word not in emission_matrix[tag]:
            if first_word.isupper():
                capital_prob = captial_matrix[tag]
            else:
                capital_prob = 1 - captial_matrix[tag]
            suffix_prob = suffix_matrix[tag]['N/A']
            for suffix in suffixes:
                if bool(re.match(r'[A-Za-z]+{}$'.format(suffix), first_word)):
                    suffix_prob = suffix_matrix[tag][suffix]
                    break
            emission_prob = capital_prob * suffix_prob
        else:
            emission_prob = emission_matrix[tag][first_word]
        state_probs[0][i] = transition_prob * emission_prob
    # run vietrbi algorithm on the remaining words
    for i in range(1, len(test_sentence)):
        for (j, tag) in enumerate(tags):
            word = test_sentence[i]
            # compare probabilities for all previous states (tags)
            combined_probs = np.array([prev_tag_prob * transition_matrix[tags[prev_tag_idx]][tag]
                                       for prev_tag_idx, prev_tag_prob in enumerate(state_probs[i - 1])])
            max_idx = np.argmax(combined_probs)
            max_prob = max(combined_probs)
            back_pointers[i][j] = max_idx
            # compute emission probability
            if word not in emission_matrix[tag]:
                # handle unknown words
                if word.isupper():
                    capital_prob = captial_matrix[tag]
                else:
                    capital_prob = 1 - captial_matrix[tag]
                suffix_prob = suffix_matrix[tag]['N/A']
                for suffix in suffixes:
                    if bool(re.match(r'[A-Za-z]+{}$'.format(suffix), word)):
                        suffix_prob = suffix_matrix[tag][suffix]
                        break
                emission_prob = capital_prob * suffix_prob
            else:
                emission_prob = emission_matrix[tag][word]
            state_probs[i][j] = max_prob * emission_prob
    tag_idx = np.argmax(state_probs[-1])
    result_tags = []
    for pointers in back_pointers[::-1]:
        result_tags.append(tags[tag_idx])
        tag_idx = pointers[tag_idx]
    result_tags.reverse()
    return (list(zip(test_sentence, result_tags)))


def run_tagger(test_data, transition_matrix, emission_matrix):
    tags = list(emission_matrix.keys())
    suffixes = ['ness', 'ship', 'ate', 'ing', 'ive', 'est',
                'al', 'ic', 'er', 'ly', 'ed', 'es', 's']
    # unknown_matrix = build_unknown_matrix(emission_matrix, tags)
    captial_matrix = build_capital_matrix(emission_matrix, tags)
    suffix_matrix = build_suffix_matrix(emission_matrix, tags, suffixes)
    tagged_test_data = []
    for test_sentence in test_data:
        tagged_test_data.append(run_tagger_on_sentence(
            test_sentence, tags, transition_matrix, emission_matrix, captial_matrix, suffix_matrix, suffixes))
    return tagged_test_data


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    matrices = read_model_file(model_file)
    transition_matrix = matrices[0]
    emission_matrix = matrices[1]
    test_data = read_test_file(test_file)
    tagged_test_data = run_tagger(
        test_data, transition_matrix, emission_matrix)
    with open(out_file, 'w') as file:
        for tagged_sentence in tagged_test_data:
            file.write(' '.join(['/'.join(pair)
                       for pair in tagged_sentence]) + '\n')
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
