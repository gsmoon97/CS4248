# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import json


def read_model_file(model_file):
    jdata = ''
    with open(model_file, 'r') as file:
        jdata = file.read()
    matrices = json.loads(jdata)
    return matrices


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    matrices = read_model_file(model_file)
    transition_matrix = matrices[0]
    emission_matrix = matrices[1]

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
