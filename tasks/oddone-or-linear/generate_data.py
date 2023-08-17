# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import shutil
import subprocess
import argparse
import os
import itertools
import random

from collections import Counter
from typing import Callable

def put_train_fpa(root: str, train_depth: int) -> None:
    """
    Saves training data to disk.
    :param root (str): the directory to save the data to.
    :param train_depth (int): the embedding depth to use for the training data.
    """
    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        for central_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
            train_input = [wrapper] * train_depth + [central_symbol] + [wrapper] * train_depth
            train_input = ' '.join(train_input)
            print(train_input, file=train_src)

            train_output = central_symbol
            print(train_output, file=train_tgt)

def put_train_mdl(root, train_depth, test_from, test_to, rule):
    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        depths = [train_depth] + [d for d in range(test_from, test_to) if d != train_depth]

        for d in depths:
            for central_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
                train_input = [wrapper] * d + [central_symbol] + [wrapper] * d
                print(' '.join(train_input), file=train_src)
                print(rule(train_input), file=train_tgt)

def put_test(root: str, depth_from: int, depth_to: int):
    """
    Saves test data to disk.
    :param root (str): the directory to save the data to to.
    :param depth_from (int): the lower bound for the embedding depth in the test data.
    :param depth_to (int): the upper bound for the embedding depth in the test data.
    """
    with open(f'{root}/test.src', 'w') as test_src, open(f'{root}/test.dst', 'w') as test_tgt:
        for d in range(depth_from, depth_to):
            for odd_one_out_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
                offsets = range(d, -d - 1, -1) if odd_one_out_symbol != wrapper else range(1)
                for offset in offsets:
                    train_input = [wrapper] * (d - offset) + [odd_one_out_symbol] + [wrapper] * (d + offset)
                    print(' '.join(train_input), file=test_src)
                    print('', file=test_tgt) # not used
    
    shutil.copy(f'{root}/test.src', f'{root}/valid.src')
    shutil.copy(f'{root}/test.dst', f'{root}/valid.dst')

def main(train_depth: int, test_span: int):
    """
    Creates a dataset of sequences of the form `a^n b a^n` to test models'
    preference for hierarchical or linear generalization.
    :param train_depth (int): the embedding depth to use in the training set.
                              corresponds to `n` in the formula above.
    :param test_span (int): test data will consist of sequences where `n` ranges
                            from train_depth - test_span to train_depth + test_span.
    """
    try:
        shutil.rmtree(str(train_depth))
    except Exception:
        pass

    pathlib.Path(str(train_depth)).mkdir()

    generate_fpa(root=f'{train_depth}/fpa/', train_depth=train_depth, test_span=test_span)
    
    # finds the nth position in a sequence
    linear = lambda seq: seq[train_depth]
    
    # finds the middle position in a sequence
    # hierar = lambda seq: seq[(len(seq) - 1) // 2]

    # finds the value that occurs least often in a sequence with two elements
    oddone = lambda seq: min(Counter(seq), key = Counter(seq).get)

    generate_mdl(root=f'{train_depth}/linear/', train_depth=train_depth, test_span=test_span, rule=linear)
    generate_mdl(root=f'{train_depth}/oddone/', train_depth=train_depth, test_span=test_span, rule=oddone)
    # generate_mdl(root=f'{train_depth}/oddone/', train_depth=train_depth, test_span=test_span, rule=odd_one_out)

def generate_fpa(root, train_depth, test_span):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)

    put_train_fpa(root_raw, train_depth)
    test_from, test_to = train_depth - test_span, train_depth + test_span + 1 # not inclusive
    put_test(root_raw, test_from, test_to)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, '--trainpref', f'{root_raw}/train', '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)

def generate_mdl(root: str, train_depth: int, test_span: int, rule: Callable):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)

    test_from, test_to = train_depth - test_span, train_depth + test_span + 1 # not inclusive

    put_test(root_raw, test_from, test_to)
    # put_test_odd_one_out(root_raw, test_from, test_to)
    put_train_mdl(root_raw, train_depth, test_from, test_to, rule)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, '--trainpref', f'{root_raw}/train', '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_depth", type=int,
                        required=True, help="Length of the training example")
    parser.add_argument("--test_span", 
                        type=int,
                        default=2,
                        )
    args = parser.parse_args()

    assert args.train_depth > 0
    main(args.train_depth, args.test_span)
