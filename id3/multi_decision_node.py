class MultiDecisionNode:

	def __init__(self, attribute, children={}):
		self.attribute = attribute
		self.children = dict(children)


	def __repr__(self):
		return self.display()


	def display(self, indent=0):
		ret = "Multi - %s" % self.attribute + ":\n"
		for child in self.children:
			ret += "  " * indent + "  %s: %s" % (child, self.children[child].display(indent+1))

		return ret


	def addBranch(self, val, node):
		self.children[val] = node
	

	def makeDecision(self, attributes={}):
		if self.attribute in attributes and attributes[self.attribute] in self.children:
			return self.children[attributes[self.attribute]].makeDecision(attributes)
		else:
			return None