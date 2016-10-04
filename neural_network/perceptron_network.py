from neural_network.perceptron import *
from neural_network.synapse import *

class PerceptronNetwork:
	eta = 0.7 # Learning gain

	def __init__(self, nb_inputs, nb_outputs, nb_hidden=None):
		if nb_hidden == None:
			nb_hidden = (nb_inputs + nb_outputs) // 2 # Mean size of input and output layers

		self.nb_inputs = nb_inputs
		self.nb_outputs = nb_outputs
		self.nb_hidden = nb_hidden
		
		self.inputs = []
		self.outputs = []
		self.hidden = []
		
		self.output_vector = []
		self.desired_outputs = None

		# Input layer construction
		for i in range(nb_inputs):
			self.inputs.append(Perceptron(0, "input"))

		# Hidden layer construction
		for h in range(nb_hidden):
			synapses = []
			for i in self.inputs:
				synapses.append(Synapse(0, i))
			
			self.hidden.append(Perceptron(0, "hidden", synapses))

		# Output layer construction
		for o in range(nb_outputs):
			synapses = []
			for h in self.hidden:
				synapses.append(Synapse(0, h))

			self.outputs.append(Perceptron(0, "output", synapses))


	def __repr__(self):
		ret = "Inputs: [\n" + "\n".join(["\t" + i.__repr__() for i in self.inputs]) + "\n]\n\n"
		ret += "Hidden: [\n" + "\n".join(["\t" + h.__repr__() for h in self.hidden]) + "\n]\n\n"
		ret += "Outputs: [\n" + "\n".join(["\t" + o.__repr__() for o in self.outputs]) + "\n]\n\n"

		return ret


	def _loadInput(self, input_vector):
		in_vect_size = len(input_vector)
		if in_vect_size == self.nb_inputs:
			for i in range(in_vect_size):
				self.inputs[i].setValue(input_vector[i])

			self._resetNodes()
		else:
			raise IndexError("Input vector size is incorrect: %d instead of %d" % (in_vect_size, self.nb_inputs))


	def _resetNodes(self):
		self.output_vector.clear()

		for h in self.hidden:
			h.reset()

		for o in self.outputs:
			o.reset()


	def predict(self, input_vector=None):
		if input_vector != None:
			self._loadInput(input_vector)
		
		if len(self.output_vector) == 0:
			for o in self.outputs:
				self.output_vector.append(o.value(True))

		return self.output_vector


	def _errorHiddenNode(self, h):
		if self.desired_outputs == None:
			raise IndexError("Desired outputs not set!")
		else:
			v = h.value()
			s = 0
			for oi in range(self.nb_outputs):
				o = self.outputs[oi]
				for syn in o.synapses:
					if syn.perceptron == h:
						s += syn.weight * self._errorOutputNode(o, self.desired_outputs[oi])

			return v * (1 - v) * s


	def _errorOutputNode(self, o, d):
		v = o.value()
		return v * (1 - v) * (d - v)


	def backPropagate(self, input_vector, desired_outputs):
		self.desired_outputs = desired_outputs
		self._loadInput(input_vector)

		# Output layer update
		for oi in range(self.nb_outputs):
			o = self.outputs[oi]
			v = o.value()
			e = self._errorOutputNode(o, desired_outputs[oi])
			for syn in o.synapses:
				syn.weight += self.eta * e * v

		# Hidden layer update
		for h in self.hidden:
			v = h.value()
			e = self._errorHiddenNode(h)
			for syn in h.synapses:
				syn.weight += self.eta * e * v

		self.desired_outputs = None


	def fit(self, training_set):
		# print("Beginning fitting:")
		# print(self)
		# print("===============================")
		# for example in training_set:
		size = len(training_set)
		for i in range(size):
			print("Fitting %d / %d" % (i+1, size))
			example = training_set[i]
			# print("Fitting example " + example.__repr__())
			self.backPropagate(example.input, example.output)
			# print(self)
			# print("----------------------------------")


	def testClassifier(self, test_set):
		nb = { True: 0, False: 0 }

		# for example in test_set:
		size = len(test_set)
		print("========================")
		for i in range(size):
			example = test_set[i]
			print("Testing %d / %d:" % (i,size))
			prediction = self.predict(example.input)
			print("\tExpected output:   " + example.output.__repr__())
			print("\tPredicted output: " + prediction.__repr__())
			print("---------------------")
			if prediction != None and prediction == example.output:
				nb[True] += 1
			else:
				nb[False] += 1

		return nb
