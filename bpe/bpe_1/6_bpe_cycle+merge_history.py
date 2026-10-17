def count_pairs(symbols):
    frequency = {}

    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
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

def select_highest_count(dictionary):
    if not dictionary:
        return None
    
    selected_pair = max(dictionary, key=dictionary.get)
    return selected_pair

def bpe_cycle(corpus, iterations, merge_history):
    if iterations <= 0:
        return corpus, merge_history
    
    print("cycle: ", iterations)

    shared_dictionary = {}
    for sublist in corpus:
        count_result = count_pairs(sublist)
        for pair, count in count_result.items():
            if pair in shared_dictionary:
                shared_dictionary[pair] += count
            else:
                shared_dictionary[pair] = count

    #print(shared_dictionary)

    selected_pair = select_highest_count(shared_dictionary)
    if selected_pair is None:
        return corpus, merge_history
    
    merge_history.append(selected_pair)
    print("merges: ", merge_history)

    new_corpus = []
    for sublist in corpus:
        word = merge(sublist, selected_pair)
        new_corpus.append(word)

    print("result:", new_corpus)

    return bpe_cycle(new_corpus, iterations -1, merge_history)
    
corpus = [
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
]

print("input")
print(corpus)

result, merges = bpe_cycle(corpus, 5, [])

print("final result: ", result, "merge history: ", merges)