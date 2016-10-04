from id3.tree_learning import *
from examples.example_set import *

# set1 = ExampleSet([
# 	Example('Attack',	{ 'health': 'healthy',	'cover': 'in_cover',	'ammo': 'with_ammo'	}),
# 	Example('Attack',	{ 'health': 'hurt',		'cover': 'in_cover',	'ammo': 'with_ammo'	}),
# 	Example('Defend',	{ 'health': 'healthy',	'cover': 'in_cover',	'ammo': 'empty'		}),
# 	Example('Defend',	{ 'health': 'hurt',		'cover': 'in_cover',	'ammo': 'empty'		}),
# 	Example('Defend',	{ 'health': 'hurt',		'cover': 'exposed',		'ammo': 'with_ammo'	}),
# 	Example('Run',		{ 'health': 'healthy',	'cover': 'exposed',		'ammo': 'with_ammo'	}),
# 	Example('Run',		{ 'health': 'healthy',	'cover': 'exposed',		'ammo': 'empty'	})
# ])

# print(set1)
# print(learningFromSet(set1))

# set2 = ExampleSet([
# 	Example('Wait',	{ 'alternate': 'Yes', 'bar': 'No' , 'fri/sat': 'No' , 'hungry': 'Yes', 'patrons': 'some', 'price': '$$$', 'rain': 'No' , 'reservation': 'Yes', 'type': 'french' , 'wait est.': '0-10'  }), # 1
# 	Example('Exit',	{ 'alternate': 'Yes', 'bar': 'No' , 'fri/sat': 'No' , 'hungry': 'Yes', 'patrons': 'full', 'price': '$'  , 'rain': 'No' , 'reservation': 'No' , 'type': 'thai'   , 'wait est.': '30-60' }), # 2
# 	Example('Wait',	{ 'alternate': 'No' , 'bar': 'Yes', 'fri/sat': 'No' , 'hungry': 'No' , 'patrons': 'some', 'price': '$'  , 'rain': 'No' , 'reservation': 'No' , 'type': 'burger' , 'wait est.': '0-10'  }), # 3
# 	Example('Wait',	{ 'alternate': 'Yes', 'bar': 'No' , 'fri/sat': 'Yes', 'hungry': 'Yes', 'patrons': 'full', 'price': '$'  , 'rain': 'Yes', 'reservation': 'No' , 'type': 'thai'   , 'wait est.': '10-30' }), # 4
# 	Example('Exit',	{ 'alternate': 'Yes', 'bar': 'No' , 'fri/sat': 'Yes', 'hungry': 'No' , 'patrons': 'full', 'price': '$$$', 'rain': 'No' , 'reservation': 'Yes', 'type': 'french' , 'wait est.': '> 60'  }), # 5
# 	Example('Wait',	{ 'alternate': 'No' , 'bar': 'Yes', 'fri/sat': 'No' , 'hungry': 'Yes', 'patrons': 'some', 'price': '$$' , 'rain': 'Yes', 'reservation': 'Yes', 'type': 'italian', 'wait est.': '0-10'  }), # 6
# 	Example('Exit',	{ 'alternate': 'No' , 'bar': 'Yes', 'fri/sat': 'No' , 'hungry': 'No' , 'patrons': 'none', 'price': '$'  , 'rain': 'Yes', 'reservation': 'No' , 'type': 'burger' , 'wait est.': '0-10'  }), # 7
# 	Example('Wait',	{ 'alternate': 'No' , 'bar': 'No' , 'fri/sat': 'No' , 'hungry': 'Yes', 'patrons': 'some', 'price': '$$' , 'rain': 'Yes', 'reservation': 'Yes', 'type': 'thai'   , 'wait est.': '0-10'  }), # 8
# 	Example('Exit',	{ 'alternate': 'No' , 'bar': 'Yes', 'fri/sat': 'Yes', 'hungry': 'No' , 'patrons': 'full', 'price': '$'  , 'rain': 'Yes', 'reservation': 'No' , 'type': 'burger' , 'wait est.': '> 60'  }), # 9
# 	Example('Exit',	{ 'alternate': 'Yes', 'bar': 'Yes', 'fri/sat': 'Yes', 'hungry': 'Yes', 'patrons': 'full', 'price': '$$$', 'rain': 'No' , 'reservation': 'Yes', 'type': 'italian', 'wait est.': '10-30' }), # 10
# 	Example('Exit',	{ 'alternate': 'No' , 'bar': 'No' , 'fri/sat': 'No' , 'hungry': 'No' , 'patrons': 'none', 'price': '$'  , 'rain': 'No' , 'reservation': 'No' , 'type': 'thai'   , 'wait est.': '0-10'  }), # 11
# 	Example('Wait',	{ 'alternate': 'Yes', 'bar': 'Yes', 'fri/sat': 'Yes', 'hungry': 'Yes', 'patrons': 'full', 'price': '$'  , 'rain': 'No' , 'reservation': 'No' , 'type': 'burger' , 'wait est.': '30-60' })  # 12
# ])

# print(set2)
# print(learningFromSet(set2))

nursery_set = ExampleSet.importFromFile('nursery.data')

def testNursery(nb):
	learning_set = ExampleSet(nursery_set.get())
	testing_set = learning_set.split(nb)
	print("Learning set: %d ; Testing set: %d" % (learning_set.size(), testing_set.size()))
	print("Learning...")
	tree = learningFromSet(learning_set)
	print("Testing...")
	res = testTree(tree, testing_set)
	print(res)
	perc = 100 * res[True] / (res[True] + res[False])
	print("Performance: %.2f %%" % perc)
	print("-----------------")

testNursery(1000)
