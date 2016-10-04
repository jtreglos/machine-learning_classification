class Synapse:

	def __init__(self, weight, perceptron):
		self.weight = weight
		self.perceptron = perceptron


	def value(self, predict=False):
		return self.perceptron.value(predict) * self.weight

