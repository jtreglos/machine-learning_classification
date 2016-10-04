from examples.example import *
import csv

class ExampleSet:

	def __init__(self, examples=[]):
		self.examples = list(examples)


	def __repr__(self):
		return "\n".join([e.__repr__() for e in self.examples])


	def attributes(self):
		if len(self.examples) != 0:
			return self.examples[0].getAttributes()
		else:
			return set([])


	def valuesForAttr(self, attr):
		return set([e.getValue(attr) for e in self.examples])

	def classifications(self):
		return set([e.classification for e in self.examples])


	def get(self):
		return self.examples


	def size(self):
		return len(self.examples)


	def split(self, nb):
		all_examples = list(self.examples)
		self.examples = all_examples[:nb]

		return ExampleSet(all_examples[nb:])


	# File should begin with attribute names on 1st line, and lines should end with classification
	@classmethod
	def importFromFile(cls, filename):
		data = []

		with open(filename, 'rt') as f:
			reader = csv.reader(f, delimiter=',', skipinitialspace=True)

			# First line is attribute names
			cols = next(reader)
			ll = len(cols)
			ci = ll - 1
			r = range(ci)

			for line in reader:
				if len(line) == ll:
					c = line[ci]
					attrs = {}
					
					for i in r:
						attrs[cols[i]] = line[i]

					data.append(Example(c, attrs))

		return cls(data)

