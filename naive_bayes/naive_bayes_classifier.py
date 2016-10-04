from helpers.helpers import *

class NaiveBayesClassifier:
	
	def __init__(self, examples, attributes=None, classifications=None):
		self.examples = examples.get()
		self.size = examples.size()

		self.class_prob_cache = {}
		self.class_sum_cache = {}
		if classifications == None:
			self.classifications = examples.classifications()
		else:
			self.classifications = classifications

		for c in self.classifications:
			self.class_sum_cache[c] = self.calcClassSum(c)
			self.class_prob_cache[c] = self.calcClassProb(c)

		if attributes == None:
			self.attributes = {}
			attributes = examples.attributes()
			for a in attributes:
				values = examples.valuesForAttr(a)
				self.attributes[a] = values
		else:
			self.attributes = attributes

		self.attr_prob_cache = {}
		for a in self.attributes:
			for v in self.attributes[a]:
				for c in self.classifications:
					key = "%s_%s_%s" % (a, v, c)
					self.attr_prob_cache[key] = self.calcAttrKnowingClassProb(a, v, c)


	def calcClassSum(self, classification):
		return sum(1 for e in self.examples if e.classification == classification)

	def calcClassProb(self, classification):
		return  self.class_sum_cache[classification] / self.size


	def calcAttrKnowingClassProb(self, attr, val, classification):
		s = sum(1 for e in self.examples if e.classification == classification and e.getValue(attr) == val)
		class_sum = self.class_sum_cache[classification]
		if class_sum != 0:
			return s / class_sum
		else:
			return 0


	def calcClassKnowingAttrsProb(self, classification, attributes):
		p = self.class_prob_cache[classification]
		
		for a in self.attributes:
			key = "%s_%s_%s" % (a, attributes[a], classification)
			p *= self.attr_prob_cache[key]

		return p


	def classify(self, attributes):
		return argMax(self.classifications, self.calcClassKnowingAttrsProb, 'classification', {'attributes': attributes})


	def testClassifier(self, example_set):
		nb = { True: 0, False: 0 }

		examples = example_set.get()
		for example in examples:
			classification = self.classify(example.attributes)
			if classification != None and classification == example.classification:
				nb[True] += 1
			else:
				nb[False] += 1

		return nb
