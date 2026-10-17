def count_pairs(symbols):
    frequency = {}
    pairs = []

    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
        pairs.append(pair)

    for pair in pairs:
        if pair in frequency:
            frequency[pair] += 1
        else:
            frequency[pair] = 1

    return frequency

def merge(sequence, selected_pair):
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