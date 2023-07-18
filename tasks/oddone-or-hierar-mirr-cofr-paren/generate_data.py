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

def put_train_fpa(root: str, n_examples_per_depth: int = 2, min_depth: int = 1, max_depth: int = 5):
    """
    Saves train data to disk.
    :param root (str): the directory to save the data to to.
    :param min_depth (int): the lower bound for the embedding depth in the test data.
    :param max_depth (int): the upper bound for the embedding depth in the test data.
    """
    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        for center_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
            for depth in range(min_depth, max_depth+1):
                n_examples = 0
                while n_examples < n_examples_per_depth:
                    train_input = ['[', center_symbol, ']']
                    while (len(train_input)/3)//2 < depth:
                        if random.random() < 0.5:
                            wrap = center_symbol
                        else:
                            wrap = wrapper
                        
                        train_input = ['[', wrap] + train_input + [wrap, ']']
                    
                    n_occurrences = Counter(train_input)
                    if (n_occurrences[center_symbol] < n_occurrences[wrapper] or center_symbol == wrapper):
                        print(' '.join(train_input), file=train_src)
                        print(center_symbol, file=train_tgt)
                        n_examples += 1

# def put_train_mdl(root, train_depth, test_from, test_to, rule):
#     with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
#         depths = [train_depth] + [d for d in range(test_from, test_to) if d != train_depth]
# 
#         for d in depths:
#             for central_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
#                 train_input = [wrapper] * d + [central_symbol] + [wrapper] * d
#                 print(' '.join(train_input), file=train_src)
#                 print(rule(train_input), file=train_tgt)

def put_test(root: str, depth_from: int, depth_to: int):
    """
    Saves test data to disk.
    :param root (str): the directory to save the data to to.
    :param depth_from (int): the lower bound for the embedding depth in the test data.
    :param depth_to (int): the upper bound for the embedding depth in the test data.
    """
    with open(f'{root}/test.src', 'w') as test_src, open(f'{root}/test.dst', 'w') as test_tgt:
        for d in range(depth_from, depth_to):
            for sequence in itertools.product(*[['a', 'b']] * (d - 1)):
                n_occurrences = Counter(sequence)
                # if the counter is len 1, that means all symbols
                # are identical. these cases are uninformative, so
                # we don't add them to the test set
                if len(n_occurrences) == 1:
                    continue
                
                if n_occurrences['a'] == n_occurrences['b']:
                    for center_symbol in ['a', 'b']:
                        out_sequence = '[ ' + ' [ '.join(sequence) + f' [ {center_symbol} ] ' + ' ] '.join(sequence)[::-1] + ' ]'
                        print(out_sequence, file=test_src)
                        print('', file=test_tgt)
                else:
                    center_symbol = n_occurrences.most_common(1)[0][0]
                    out_sequence = '[ ' + ' [ '.join(sequence) + f' [ {center_symbol} ] ' + ' ] '.join(sequence)[::-1] + ' ]'
                    print(out_sequence, file=test_src)
                    print('', file=test_tgt)
    
    shutil.copy(f'{root}/test.src', f'{root}/valid.src')
    shutil.copy(f'{root}/test.dst', f'{root}/valid.dst')

def main(test_depth: int, test_span: int, train_n_examples_per_depth: int, train_min_depth: int = 1, train_max_depth: int = 5):
    """
    Creates a dataset of sequences of the form `a^n b a^n` to test models'
    preference for hierarchical or linear generalization using a PCFG.
    :param test_depth (int): the embedding depth to use in the test set.
                             corresponds to `n` in the formula above.
    :param test_span (int): generate test examples where n is from test_depth +/- test_span.
    :param train_n_examples_per_depth (int): generate n examples per combination of 
                                             wrapper and center symbol in train data.
	:param train_min_depth (int): the minimum allowable recursion depth in train sentences.
    :param train_max_depth (int): the maximum allowable recursion depth in train sentences.
    """
    try:
        shutil.rmtree(str(test_depth))
    except Exception:
        pass

    pathlib.Path(str(test_depth)).mkdir()

    generate_fpa(
        root=f'{test_depth}/fpa/', 
        test_depth=test_depth,
        test_span=test_span,
        train_n_examples_per_depth=train_n_examples_per_depth,
		train_min_depth=train_min_depth,
        train_max_depth=train_max_depth,
    )
    
    # finds the nth position in a sequence
    # linear = lambda seq: seq[train_depth]
    
    # finds the middle position in a sequence
    # hierar = lambda seq: seq[(len(seq) - 1) // 2]

    # generate_mdl(root=f'{train_depth}/linear/', train_depth=train_depth, test_p_recursion=test_p_recursion, test_n_examples_per_combination=test_n_examples_per_combination, text_max_depth=test_max_depth, rule=linear)
    # generate_mdl(root=f'{train_depth}/hierar/', train_depth=train_depth, test_p_recursion=test_p_recursion, test_n_examples_per_combination=test_n_examples_per_combination, text_max_depth=test_max_depth, rule=hierar)

def generate_fpa(root, test_depth, test_span, train_n_examples_per_depth, train_min_depth: int = 1, train_max_depth: int = 5):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)

    put_train_fpa(root_raw, train_n_examples_per_depth, train_min_depth, train_max_depth)
    test_from, test_to = test_depth - test_span, test_depth + test_span + 1 # not inclusive
    put_test(root_raw, test_from, test_to)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, '--trainpref', f'{root_raw}/train', '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)

# def generate_mdl(root: str, train_depth: int, test_p_recursion: float, test_n_examples_per_combination: int, test_max_depth: int = 0, rule: Callable):
#     root_raw = f'{root}/data/'
#     pathlib.Path(root_raw).mkdir(parents=True)
# 
#     put_test(root_raw, test_p_recursion, test_n_examples_per_combination, test_max_depth)
#     put_train_mdl(root_raw, train_depth, test_from, test_to, rule)
# 
#     root_bin = f'./{root}/data-bin/'
#     pathlib.Path(root_bin).mkdir(parents=True)
# 
#     command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
#                'dst', '--destdir', root_bin, '--trainpref', f'{root_raw}/train', '--testpref', f'{root_raw}/test']
# 
#     subprocess.check_call(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_depth", type=int,
                        required=True, 
                        help="Length of the test example +/- test_span"
                        )
    parser.add_argument("--test_span", 
                        type=int,
                        default=2,
                        )
    parser.add_argument("--train_n_examples_per_depth",
                        type=int,
                        default=25,
                        )
    parser.add_argument("--train_min_depth",
                        type=int,
                        default=1,
                        )
    parser.add_argument("--train_max_depth",
                        type=int,
                        default=5,
                        )
    args = parser.parse_args()

    assert args.test_depth > 0
    main(**vars(args))
