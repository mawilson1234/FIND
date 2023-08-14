import re
import os
import sys
import gzip
import random
import itertools

from tqdm import tqdm
from typing import *

from nltk import CFG, Tree, nonterminals
from nltk.grammar import Nonterminal

C, S, V, D1, D2, D, U = nonterminals(
	'C, S, V, D1, D2, D, U'
)

scan_grammar = CFG.fromstring("""
	C -> S | S 'after' S | S 'and' S
	S -> V | V 'twice' | V 'thrice'
	V -> U | D | D1 'opposite' D2 | D1 'around' D2
	D1 -> 'turn' | U
	D2 -> 'left' | 'right'
	D -> 'turn left' | 'turn right' | U 'left' | U 'right'
	U -> 'walk' | 'look' | 'run' | 'jump'
""")

TERMINAL_DENOTATIONS: Dict[str, str] = {
	'walk': 'I_WALK',
	'look': 'I_LOOK',
	'run': 'I_RUN',
	'jump': 'I_JUMP',
	'turn': '',
	'turn left': 'I_TURN_LEFT',
	'turn right': 'I_TURN_RIGHT',
	'left': 'I_TURN_LEFT',
	'right': 'I_TURN_RIGHT',
}

def trees(
	grammar: CFG, 
	start: Nonterminal = None, 
	depth: int = sys.maxsize, 
	n: int = None
):
	"""
	Generates all trees from a CFG.
	
	:param grammar: The Grammar used to generate sentences.
	:param start: The Nonterminal from which to start generate sentences.
	:param depth: The maximal depth of the generated tree.
	:param n: The maximum number of sentences to return.
	:return: An iterator of lists of terminal tokens.
	"""
	if not start:
		start = grammar.start()
	
	iter = _generate_all(grammar, [start], None, True, depth)
	
	if n:
		iter = itertools.islice(iter, n)
	
	return iter

def _generate_all(
	grammar: CFG, 
	items: List, 
	parent: Union[Nonterminal, str] = None, 
	is_left_edge: bool = True, 
	depth: int = sys.maxsize
) -> Tree:
	if items:
		for frag1 in _generate_one(grammar, items[0], parent, is_left_edge = True, depth = depth):
			for frag2 in _generate_all(grammar, items[1:], parent, is_left_edge = False, depth = depth):
				if parent is not None and is_left_edge:
					yield [Tree(parent, frag1 + frag2)]
				elif parent is None and is_left_edge:
					yield (frag1 + frag2)[0]
				else:
					yield frag1 + frag2
	else:
		yield []

def _generate_one(
	grammar: CFG,
	item: List[str],
	parent: Union[Nonterminal,str] = None,
	is_left_edge: bool = True,
	depth: int = None,
) -> Tree:
	if depth > 0:
		if isinstance(item, Nonterminal):
			for prod in grammar.productions(lhs=item):
				yield from _generate_all(
					grammar = grammar, 
					items = prod.rhs(), 
					parent = prod.lhs(), 
					is_left_edge = is_left_edge, 
					depth = depth - 1
				)
		else:
			yield [item]

def get_labels(
	tree: Tree,
	as_strings: bool = False
) -> List:
	'''
	Returns a list of top-level labels
	for a tree.
	
	:param tree: the tree to return the top-level node labels for.
	:param as_strings: whether to return the node labels as strings or non-terminals.
	'''
	labels = []
	for item in tree:
		if isinstance(item, str):
			labels.append(item)
			continue
		
		if as_strings:
			labels.append(item.label().symbol())
			continue
		
		labels.append(item.label())
	
	return labels

def denotation(tree: Tree) -> str:
	'''
	Convert a tree to its SCAN denotation (i.e., the corresponding output string).
	'''
	
	labels = get_labels(tree)
	if all(label in TERMINAL_DENOTATIONS for label in labels):
		return [TERMINAL_DENOTATIONS[label] for label in labels]
	
	if labels == [S, 'after', S]:
		return denotation(tree[2]) + denotation(tree[0])
	
	if labels == [S, 'and', S]:
		return denotation(tree[0]) + denotation(tree[2])
		
	if labels == [V, 'thrice']:
		return denotation(tree[0]) * 3
	
	if labels == [V, 'twice']:
		return denotation(tree[0]) * 2
	
	if labels == [D1, 'around', D2]:
		return (
			denotation(tree[2]) + denotation(tree[0]) +
			denotation(tree[2]) + denotation(tree[0]) +
			denotation(tree[2]) + denotation(tree[0]) +
			denotation(tree[2]) + denotation(tree[0])
		)
	
	if labels == [D1, 'opposite', D2]:
		return denotation(tree[2]) + denotation(tree[2]) + denotation(tree[0])
	
	if labels == [U, 'left'] or labels == [U, 'right']:
		return denotation([tree[1]]) + denotation(tree[0])
	
	if len(labels) == 1:
		return denotation(tree[0])
	
	breakpoint()
	raise ValueError(f'Unable to determine a parse for {labels!r}.')

def SCAN_generator(
	filter: Callable = lambda source: source,
) -> None:
	'''
	Generates input-output pairs from the grammar matching
	a filter.
	'''
	for tree in trees(scan_grammar):
		source = filter(tree)
		if source:
			target = ' '.join(denotation(source)).strip()
			while '  ' in target:
				target = target.replace('  ', ' ')
			
			source = ' '.join(source.leaves())
			yield {'IN': source, 'OUT': target}

def save_SCAN(
	save_dir: str = '', 
	splits: List[Dict] = [{'name': 'all', 'filter': lambda source, target: True, 'shuffle': True}]
) -> None:
	# pre-create the files so we can append later
	os.makedirs(save_dir, exist_ok=True)
	
	files = [f'tasks_{name}.txt' for name in [d['name'] for d in splits]]
	for file in files:
		with open(os.path.join(save_dir, file), 'wt') as out_file:
			_ = out_file.write('')	
	
	for tree in tqdm(trees(scan_grammar)):
		for split in splits:
			source = split['filter'](tree)
			if source:
				target = ' '.join(denotation(source)).strip()
				source = ' '.join(source.leaves())
				while '  ' in target:
					target = target.replace('  ', ' ')
				
				line = f'IN: {source} OUT: {target}\n'
				with open(os.path.join(save_dir, f'tasks_{split["name"]}.txt'), 'at') as out_file:
					_ = out_file.write(line)
	
	print('Compressing output files.')
	for split in splits:
		with open(os.path.join(save_dir, f'tasks_{split["name"]}.txt'), 'rt') as in_file:
			data = in_file.readlines()
		
		if split.get('shuffle', False):
			random.shuffle(data)
		
		with gzip.open(os.path.join(save_dir, f'tasks_{split["name"]}.txt.gz'), 'wt') as out_file:
			for line in data:
				_ = out_file.write(line)
		
		os.remove(os.path.join(save_dir, f'tasks_{split["name"]}.txt'))

def train_addprim_jump(source: Tree) -> Tree:
	'''
	Returns source if (i) jump is not in source or (ii) only jump is in source.
	Else, returns a tree with just 'jump'
	'''
	if source.leaves() == ['jump'] or ['jump'] not in source.leaves():
		return source
	
	return Tree(U, ['jump'])

def test_addprim_jump(source: Tree) -> Tree:
	'''
	Returns source if (i) jump is in source and (ii) jump is not all in source.
	Else, returns None
	'''
	if 'jump' in source.leaves() and not source.leaves() == ['jump']:
		return source

def train_addtwicethrice_jump(source: Tree) -> Tree:
	'''
	Returns source if neither twice nor thrice scope over jump in source, else returns 'jump'.
	'''
	# easy, no jump means no scoping to figure out
	if 'jump' not in source.leaves():
		return source
	
	# same deal
	if not any(x in ['twice', 'thrice'] for x in source.leaves()):
		return source
	
	scope_positions = [position for position in source.treepositions() if source[position] in ['twice', 'thrice']]
	for position in scope_positions:
		# twice/thrice always occur immediately to the right of what they scope over,
		# so we change the final index to 0 to find their scope
		v_position = tuple([*position[:-1], 0])
		if 'jump' in source[v_position].leaves():
			return Tree(U, ['jump'])
	
	return source

def test_addtwicethrice_jump(source: Tree) -> Tree:
	if train_addtwicethrice_jump(source) == Tree(U, ['jump']):
		return source
	
	return ''

if __name__ == '__main__':
	# demo
	# splits = [
	# 	dict(
	# 		name = 'train_addprim_jump',
	# 		filter = lambda source: source if 'jump' not in source or source == 'jump' else 'jump',
	# 		shuffle = True,
	# 	),
	# 	dict(
	# 		name = 'test_addprim_jump',
	# 		filter = lambda source: source if 'jump' in source and source != 'jump' else '',
	# 	),
	# ]
	splits = [
		dict(
			name = 'train_addtwicethrice_jump',
			filter = train_addtwicethrice_jump,
			shuffle = True,
		),
		dict(
			name = 'test_addtwicethrice_jump',
			filter = test_addtwicethrice_jump,
		)
	]
	
	save_SCAN(save_dir='.', splits=splits)
