import sys

def make_equation(coordinates, characters):
	"""
	Takes coordinates and characters and returns equation they make
	Sorts them in the right order depending on x coordinate
	:param coordinates: np.array
		(x,y) coordinates of characters
	:param characters: list
		list of characters
	:return equation: string
		equation made from characters
	"""
	# get x coordinates from tuples
	x_coordinates = [c[1] for c in coordinates]

	# pair them up with characters in a dictionary
	x_c_pairs = {x:c for x,c in zip(x_coordinates, characters)}

	# construct the equation by sorting x coordinates in ascending order
	equation = ""
	for x in sorted(x_c_pairs):
		equation += x_c_pairs[x] + ' '
	return equation


def solve(equation):
	"""
	:param equation: string
		equation to be solved
	:return: float
		result of equation
	"""
	# solve everything inside of brackets first and put it back in list
	if '(' in equation:
		# inside is index of first bracket and index of last bracket
		index_of_first_bracket = equation.index('(')
		index_of_last_bracket = len(equation) - 1 - equation[::-1].index(')')

		# replace it with solved expression
		equation = equation.replace(equation[index_of_first_bracket:index_of_last_bracket+1],
		                 str(solve(equation[index_of_first_bracket+1:index_of_last_bracket])))


	# solve equation
	current_number = 0
	current_operator = ''
	for c in equation:
		if c.isdigit():
			if not current_operator:
				current_number += float(c)
			else:
				if current_operator=='+':
					current_number += float(c)
				elif current_operator=='-':
					current_number -= float(c)
		else:
			current_operator = c

	return current_number