from math import log
from examples.example import *
from id3.leaf_node import *
from id3.multi_decision_node import *
from helpers.helpers import *


LOG_OF_2 = log(2)


def classificationCount(examples):
	c = [e.classification for e in examples]
	return {i:c.count(i) for i in c}


def pluralityValue(examples):
	d = classificationCount(examples)
	k,v = list(d.keys()), list(d.values())

	return k[v.index(max(v))]


def distinctClassifications(examples):
	return set([e.classification for e in examples])


def distinctAttrValues(a, examples):
	return set([e.getValue(a) for e in examples])


def log2(v):
	return log(v) / LOG_OF_2


def entropy(examples):
	e = 0
	cc = classificationCount(examples)
	nb_total = sum(list(cc.values()))
	
	for c in cc:
		p = cc[c] / nb_total
		e -= p * log2(p)

	return e


def subset(a, val, examples):
	return [e for e in examples if e.getValue(a) == val]


def conditionalEntropy(a, examples):
	sum = 0
	values = distinctAttrValues(a, examples)
	nb = len(examples)
	
	if nb != 0:
		for val in values:
			si = subset(a, val, examples)
			sum += (len(si) / nb) * entropy(si)

		return sum
	else:
		return 0


def informationGain(a, examples):
	return entropy(examples) - conditionalEntropy(a, examples)


def intrinsicValue(a, examples):
	sum = 0
	values = distinctAttrValues(a, examples)
	nb = len(examples)

	if nb != 0:
		for val in values:
			si = subset(a, val, examples)
			d = len(si) / nb
			sum += d * log2(d)

		return -sum
	else:
		return 0


def informationGainRatio(a, examples):
	iv = intrinsicValue(a, examples)
	ig = informationGain(a, examples)
	if iv != 0:
		return ig / iv
	else:
		return 0


def decisionTreeLearning(examples, attributes, parent_examples=[]):
	if len(examples) == 0:
		return LeafNode(pluralityValue(parent_examples))
	else:
		classifications = distinctClassifications(examples)
		if len(classifications) == 1:
			return LeafNode(classifications.pop())
		elif attributes == None or len(attributes) == 0:
			return LeafNode(pluralityValue(examples))
		else:
			attr = argMax(attributes, informationGain, 'a', {'examples': examples})
			tree = MultiDecisionNode(attr)
			values = distinctAttrValues(attr, examples)
			for val in values:
				exs = subset(attr, val, examples)

				remaining_attributes = attributes.copy()
				remaining_attributes.remove(attr)

				subtree = decisionTreeLearning(exs, remaining_attributes, examples)
				tree.addBranch(val, subtree)

			return tree


def learningFromSet(example_set):
	return decisionTreeLearning(example_set.get(), example_set.attributes())


def testTree(tree, example_set):
	nb = { True: 0, False: 0 }

	examples = example_set.get()
	for example in examples:
		decision = tree.makeDecision(example.attributes)
		if decision != None and decision.classification == example.classification:
			nb[True] += 1
		else:
			nb[False] += 1

	return nb
