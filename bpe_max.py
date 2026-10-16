pair_list = {('b', 'a'): 1, ('a', 'n'): 2, ('n', 'a'): 2}

highest_count = max(pair_list, key=pair_list.get)

print (highest_count)