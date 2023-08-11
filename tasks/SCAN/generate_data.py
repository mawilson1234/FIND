# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import shutil
import pathlib
import argparse
import subprocess

from tqdm import tqdm
from generate_SCAN import *

SPLITS = {
	'addtwicethrice_jump': {
		'train': train_addtwicethrice_jump,
		'test': test_addtwicethrice_jump,
	},
}

def put_train_fpa(root: str, split: str, shuffle: bool = False) -> None:
	'''
	Saves training data to disk.
	:param root (str): the directory to save the data to.
	:param split (str): identifies the functions to use for saving train data in SPLITS.
	:param shuffle (bool): whether to shuffle the dataset after generating it.
	'''
	filter_function = SPLITS.get(split, {}).get('train', lambda x: x)
	
	sources = []
	targets = []
	print('Generating train examples.')
	for example in tqdm(SCAN_generator(filter=filter_function)):
		sources.append(example['IN'])
		targets.append(example['OUT'])
	
	if shuffle:
		zipped = list(zip(sources, targets))
		random.shuffle(zipped)
		sources = [source for source, _ in zipped]
		targets = [target for _, target in zipped]
	
	with open(os.path.join(f'{root}', 'train.src'), 'wt') as train_src, \
		 open(os.path.join(f'{root}', 'train.dst'), 'wt') as train_tgt:
		for source, target in zip(sources, targets):
			print(source, file=train_src)
			print(target, file=train_tgt)

def put_test(root: str, split: str):
	'''
	Saves test data to disk.
	:param root (str): the directory to save the data to to.
	:param split (str): identifies the function to use for saving test data in SPLITS.
	'''
	filter_function = SPLITS.get(split, {}).get('test', lambda x: x)
	
	print('Generating test examples.')
	with open(os.path.join(f'{root}', 'test.src'), 'wt') as test_src, \
		 open(os.path.join(f'{root}', 'test.dst'), 'wt') as test_tgt:
		for example in tqdm(SCAN_generator(filter=filter_function)):
			print(example['IN'], file=test_src)
			print(example['OUT'], file=test_tgt) # used for eval
	
	shutil.copy(os.path.join(f'{root}', 'test.src'), os.path.join(f'{root}', 'valid.src'))
	shutil.copy(os.path.join(f'{root}', 'test.dst'), os.path.join(f'{root}', 'valid.dst'))

def generate_fpa(root: str, split: str, shuffle_train: bool) -> None:
	root_raw = os.path.join(f'{root}', 'data')
	pathlib.Path(root_raw).mkdir(parents=True)

	put_train_fpa(root_raw, split, shuffle_train)
	put_test(root_raw, split)
	
	root_bin = os.path.join(f'{root}', 'data-bin')
	pathlib.Path(root_bin).mkdir(parents=True)
	
	command = [
		'fairseq-preprocess', 
		'--source-lang', 'src', 
		'--target-lang', 'dst', 
		'--destdir', root_bin, 
		'--trainpref', os.path.join(f'{root_raw}', 'train'), 
		'--testpref', os.path.join(f'{root_raw}', 'test'),
	]
	
	subprocess.check_call(command)

def main(split: str, shuffle_train):
	'''
	Creates a dataset of sequences of the form `a^n b a^n` to test models'
	preference for hierarchical or linear generalization.
	:param split: name of the function to use to generate the train/test splits.
	'''
	try:
		shutil.rmtree(split)
	except Exception:
		pass
	
	pathlib.Path(split).mkdir()
	
	generate_fpa(root=os.path.join(f'{split}', 'fpa'), split=split, shuffle_train=shuffle_train)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--split', 
		type=str,
		required=True, 
		help='Name of the function to use to generate train/test splits'
	)
	parser.add_argument(
		'--shuffle-train',
		action='store_true',
		help='Whether to shuffle the training set (recommended).'
	)
	args = parser.parse_args()
	
	main(args.split, args.shuffle_train)
