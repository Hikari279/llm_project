def merge(sequence, selected_pair):
    if len(sequence) == 0: 
        return []
    
    new_list = []

    i = 0
    while i < len(sequence):

        if i + 1 < len(sequence):
            pair = sequence[i], sequence[i + 1]
            if pair == selected_pair:
                new_list.append(sequence[i] + sequence[i+1])
                i += 2
                continue
            
        new_list.append(sequence[i])
        i += 1

    return new_list

symbols = []
chosen_pair = ('a', 'b')

result = merge(symbols, chosen_pair)
print(result)