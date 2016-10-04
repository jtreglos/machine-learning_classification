class LeafNode:

	def __init__(self, classification):
		self.classification = classification


	def __repr__(self):
		return self.display()


	def display(self, indent=0):
		return "Leaf - %s" % self.classification + "\n"


	def makeDecision(self, attributes={}):
		return self