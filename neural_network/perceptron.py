import math

class Perceptron:

	h = 1.0 # Sigmoid function coefficient


	def __init__(self, bias_or_val=0, ptype="hidden", synapses=[]):
		self.ptype = ptype
		
		if ptype == "input":
			self.val = bias_or_val
			self.synapses = []
			self.bias = 0
		else:
			self.synapses = synapses
			self.bias = bias_or_val
			self.val = None


	def __repr__(self):
		if self.ptype == "input":
			return "%s: %.2f" % (self.ptype, self.val)
		else:
			if self.val == None:
				return ("%s: (%.2f / " % (self.ptype, self.bias)) + ", ".join(["%.2f" % s.weight for s in self.synapses]) + ") -> -"
			else:	
				return ("%s: (%.2f / " % (self.ptype, self.bias)) + ", ".join(["%.2f" % s.weight for s in self.synapses]) + ") -> " + "%.2f" % self.val


	def setValue(self, value):
		self.val = value


	def _sigmaLearn(self, val):
		return 1.0 / (1.0 + math.exp(-self.h * val))


	def _sigmaPredict(self, val):
		if val <= 0:
			return 0
		else:
			return 1


	def reset(self):
		self.val = None


	def value(self, predict=False):
		if self.ptype != "input" and self.val == None:
			if predict:
				self.val = self._sigmaPredict(self.bias + sum([s.value(predict) for s in self.synapses]))
			else:
				self.val = self._sigmaLearn(self.bias + sum([s.value(predict) for s in self.synapses]))

		return self.val
