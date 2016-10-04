class Example:

	def __init__(self, classification, attributes):
		self.classification = classification
		self.attributes = dict(attributes)


	def __repr__(self):
		return "(" + ", ".join([a+": "+self.attributes[a] for a in self.attributes]) + ") -> " + self.classification



	def getValue(self, attribute):
		return self.attributes.get(attribute, None)


	def getAttributes(self):
		return set(self.attributes.keys())
