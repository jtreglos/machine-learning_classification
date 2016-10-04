from examples.example_set import *
from naive_bayes.naive_bayes_classifier import *

# learning_set = ExampleSet([
# 	Example('Brake',	{ 'distance': 'near',	'speed': 'slow' }),
# 	Example('Brake',	{ 'distance': 'near',	'speed': 'fast' }),
# 	Example('Brake',	{ 'distance': 'far' ,	'speed': 'fast' }),
# 	Example('Brake',	{ 'distance': 'far' ,	'speed': 'slow' }),
# 	Example('Brake',	{ 'distance': 'near',	'speed': 'fast' }),
# 	Example('No Brake',	{ 'distance': 'far' ,	'speed': 'fast' }),
# 	Example('No Brake',	{ 'distance': 'near',	'speed': 'slow' })
# ])

# test_attributes = { 'distance': 'far', 'speed': 'slow' }

# nbc = NaiveBayesClassifier(learning_set)
# print(nbc.classify(test_attributes))



learning_set = ExampleSet([
	Example('male'  , { 'height': 6.00, 'weight': 180, 'foot_size': 12 }),
	Example('male'  , { 'height': 5.92, 'weight': 190, 'foot_size': 11 }),
	Example('male'  , { 'height': 5.58, 'weight': 170, 'foot_size': 12 }),
	Example('male'  , { 'height': 5.92, 'weight': 165, 'foot_size': 10 }),
	Example('female', { 'height': 5.00, 'weight': 100, 'foot_size':  6 }),
	Example('female', { 'height': 5.50, 'weight': 150, 'foot_size':  8 }),
	Example('female', { 'height': 5.42, 'weight': 130, 'foot_size':  7 }),
	Example('female', { 'height': 5.75, 'weight': 150, 'foot_size':  9 })
])

test_attributes = { 'height': 6.00, 'weight': 130, 'foot_size': 8 }

nbc = NaiveBayesClassifier(learning_set)
print(nbc.classify(test_attributes))



# nursery_set = ExampleSet.importFromFile('nursery.data')
# nursery_attributes = {
# 	'children': { '1', '2', '3', 'more' },
#  	'finance': {'convenient', 'inconv'},
#  	'form': {'complete', 'completed', 'foster', 'incomplete'},
#  	'has_nurs': {'critical', 'improper', 'less_proper', 'proper', 'very_crit'},
#  	'health': {'not_recom', 'priority', 'recommended'},
#  	'housing': {'convenient', 'critical', 'less_conv'},
#  	'parents': {'great_pret', 'pretentious', 'usual'},
#  	'social': {'nonprob', 'problematic', 'slightly_prob'}
#  }
# nursery_classifications = { 'not_recom', 'priority', 'recommend', 'spec_prior', 'very_recom' }

# def testNursery(nb):
# 	learning_set = ExampleSet(nursery_set.get())
# 	testing_set = learning_set.split(nb)
# 	print("Learning set: %d ; Testing set: %d" % (learning_set.size(), testing_set.size()))
# 	print("Loading...")
# 	nbc = NaiveBayesClassifier(learning_set, nursery_attributes, nursery_classifications)
# 	print("Testing...")
# 	res = nbc.testClassifier(testing_set)
# 	print(res)
# 	perc = 100 * res[True] / (res[True] + res[False])
# 	print("Performance: %.2f %%" % perc)
# 	print("-----------------")

# 	return nbc

# print("Syntax: testNursery(size_learning_set)")
