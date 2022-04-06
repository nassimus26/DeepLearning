def max_result_expression(expression, variables):
    print()
    print(expression, variables)

    # request is to find the expression that has the max value
    # to help us, let's put the expression into a list
    # later we will iterate through this list "expression_list"
    # by passing the expression to our earlier code
    # the result of each expression will be stored into another list
    # let's call that result list as "expression_results" list

    expression_list = [expression]

    expression_results = []

    # Now that we have assigned the two lists, let's review the variables being passed

    # first check if the variables string is a dictionary or a string
    # isinstance() allows us to check for it
    # isinstance(variable_name, datatype). datatype can be int, str, float, list, tuple, dict
    # here we are checking if the variable is a dictionary (dict)
    # isinstance() will return True or False. In this case, we will get True if variable is a dictionary

    if isinstance(variables, dict):

        # since the variable is a dictionary, we need to iterate through it like a dictionary
        # use key,value to get the key and value portion

        for k, v in variables.items():

            # the value portion of the dictionary can be either an integer or a tuple
            # check if the value is a tuple. If tuple, then we have to do a range

            if isinstance(v, tuple):

                # since the value is a tuple, we need to do a range of (lower_limit, upper_limit)
                # one expression will now become multiple expressions

                # example: x+y with x:(2,5) will become 2+y,3+y,4+y (3 expressions)
                # store each expression into a list. To do that, create a temp_exp list

                temp_exp = []

                # iterate through the original expression_list (which started with one expression)

                for exp_temp in expression_list:

                    # for each expression in the expression_list,
                    # let's replace the value of variable with range value
                    for x in range(int(v[0]), int(v[1])):
                        # we can do that using string replace

                        temp_exp.append(exp_temp.replace(k, str(x)))

                # now that we have replaced all the expressions
                # with the range value, let's reassign the temp_exp back to expression_list
                # Note: If there are multiple variables with ranges, this happens again
                expression_list = temp_exp[:]

    # finally, we have a list of expressions that need to be evaluated
    # let's iterate through the expression one at a time

    # As I mentioned earlier, store the result of the expression
    # in the expression_results list
    # finally, get the max value from this list and return it back

    for expression in expression_list:
        stack = []
        e_list = [x for x in expression.split(' ')[::-1] if x != '']

        # if e_list has only one numeric value, then return numeric value
        # if e_list has only one value but not numeric, then return null

        if len(e_list) == 1 and e_list[0].isdigit():
            expression_results.append(int(e_list[0]))
            continue

        elif len(e_list) == 1 and not e_list[0].isdigit():
            expression_results.append('null')
            continue

        # if there are less than 3 arguments in an expression
        # force the result as null

        elif len(e_list) < 3:
            expression_results.append('null')
            continue

        for c in e_list:
            if c.isdigit():
                stack.append(int(c))
            elif c.isalpha():
                stack.append(int(variables[c]))
            else:
                if len(stack) > 1:
                    o1 = stack.pop()
                    o2 = stack.pop()

                    if c == '+':
                        stack.append(o1 + o2)
                    elif c == '-':
                        stack.append(o1 - o2)
                    elif c == '*':
                        stack.append(o1 * o2)
                    elif c == '/':
                        stack.append(o1 / o2)

        expression_results.append(stack.pop() if len(stack) == 1 else 'null')

    return max(expression_results)


print(max_result_expression('+ 6 * - 4 + 2 3 8', ''))
print(max_result_expression('* * * x -1 -1 -1', {'x' :(2, 6)}))

## print(max_result_expression('+ 10  20 50 ', ''))

## print(max_result_expression('* + 1 2 3', ''))

## print(max_result_expression('* + 2 x y', {'x': 1, 'y': 3}))

## print(max_result_expression('* + 2 x y', {'x': (3, 7), 'y': (1, 9)}))

## print(max_result_expression('* + 2 x y', {'x': (0, 2), 'y': (2, 4)}))

## print(max_result_expression('+ 6 * - x + 2 3 8', {'x': (4, 8), 'y': (2, 4)}))