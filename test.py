number_list = [1, 2, 3, 3]
str_list = ['one', 'two', 'three', 'three']

result = zip(number_list)
print(list(result))

# Two iterables are passed
result = zip(number_list, range(len(number_list)))
print(dict(result))

# Converting iterator to set
#result_set = set(result)
#print(result_set)