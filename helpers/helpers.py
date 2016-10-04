def argMax(attributes, func, attr_name, add_params={}):
	best_attr = None
	best_val = 0
	for attr in attributes:
		params = add_params.copy()
		params[attr_name] = attr
		new_val = func(**params)
		if new_val > best_val:
			best_val = new_val
			best_attr = attr

	return best_attr