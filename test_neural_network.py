from neural_network.perceptron_network import *
from mnist.loader import MNIST

class SimpleExample:

	def __init__(self, input, output):
		self.input = input
		self.output = [0] * 10
		self.output[output] = 1


	def __repr__(self):
		return self.input.__repr__() + (" => " + self.output.__repr__())


# set1 = [
# 	SimpleExample([1, 1, 1], 1),
# 	SimpleExample([0, 1, 1], 1),
# 	SimpleExample([1, 1, 0], 0),
# 	SimpleExample([0, 1, 0], 0),
# 	SimpleExample([0, 0, 1], 0)
# ]

# pn = PerceptronNetwork(3, 1)
# for i in range(100):
# 	pn.fit(set1)
# print(pn.testClassifier(set1))


mndata = MNIST(".")
# mndata.load_training()
mndata.load_testing()

examples = []
# size = len(mndata.test_images)
size = 50
for i in range(size):
	inpt = mndata.test_images[i]
	outpt = mndata.test_labels[i]
	example = SimpleExample(inpt, outpt)
	examples.append(example)

pn = PerceptronNetwork(784, 10, 300)
pn.fit(examples)
print(pn.testClassifier(examples))
