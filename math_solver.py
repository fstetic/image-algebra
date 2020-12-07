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
	x_coordinates = [c[0] for c in coordinates]

	# pair them up with characters in a dictionary
	x_c_pairs = {x:c for x,c in zip(x_coordinates, characters)}

	# construct the equation by sorting x coordinates in ascending order
	equation = ""
	for x in sorted(x_c_pairs):
		equation += x_c_pairs[x] + ' '

	# remove the last space
	return equation[:-1]


def solve(equation):
	"""
	Solves equation using shunting-yard algorithm
	:param equation: string
		equation to be solved
	:return: float
		result of equation
	"""
	postfix = rpn(equation)
	result = shunting_yard(postfix)
	return result


def rpn(equation):
	"""
	Takes equation and returns reverse polish notation for it
	i.e. performs infix to postfix conversion
	:param equation: string
		equation to parse
	:return: list
		rpn of equation
	"""
	precedence = {'+':1, '-':1, '*':2, '/':2, '(':3}    # needed to respect the order of operations
	operators = ['+','-','*','/']   # needed for error handling
	operator_stack, postfix_result = list(), list()
	elements = equation.split(' ')

	# look if expression ends in an operator
	found_operator = False
	for e in elements[::-1]:
		if e.isdigit():
			# if there's an operator after the last digit
			if found_operator:
				raise_error("Unterminated expression", equation)
			else:
				break
		elif e in operators:
			found_operator = True

	for e in elements:
		if e.isdigit():
			# just add digits
			postfix_result.append(e)
		elif e == '(':
			# start the operands in brackets
			operator_stack.append(e)
		elif e == ')':
			top_operand = operator_stack.pop()
			# while inside the bracket
			while top_operand != '(':
				# add all operands
				postfix_result.append(top_operand)
				top_operand = operator_stack.pop()
		else:
			if e not in operators:
				raise_error("Unsupported character", equation)
			# while there are more important operands on stack, add them before current
			while operator_stack and precedence[operator_stack[-1]] >= precedence[e] and operator_stack[-1] != '(':
				postfix_result.append(operator_stack.pop())
			operator_stack.append(e)

	# empty the stack
	while operator_stack:
		# if there are brackets at this point it means some were not opened or closed
		if '(' in operator_stack or ')' in operator_stack:
			raise_error("Missing brackets", equation)
		postfix_result.append(operator_stack.pop())

	return postfix_result


def shunting_yard(elements):
	"""
	Performs shunting-yard on a list of characters
	:param elements: list
		equation characters in reverse polish notation
	:return: float
		result of equation
	"""
	# init dict with operations mapping
	operations = {
		'+': lambda x, y: x + y,
		'-': lambda x, y: x - y,
		'*': lambda x, y: x * y,
		'/': lambda x, y: x / y,
	}
	operand_stack = list()

	for e in elements:
		if e.isdigit():
			operand_stack.append(e)
		else:
			# take operands from the top of the stack
			operand2 = float(operand_stack.pop())
			operand1 = float(operand_stack.pop()) if operand_stack else 0.0 # support for negative numbers
			# perform operation
			result = operations[e](operand1, operand2)
			# return result to the stack
			operand_stack.append(result)

	# last element on top of the stack is the result
	return operand_stack.pop()


def raise_error(message, equation):
	"""
	Raises error and exits execution
	:param message: string
		message to display
	:param equation: string
		equation in which the error occurred
	"""
	print(message,"in equation", equation, file=sys.stderr)
	exit(1)