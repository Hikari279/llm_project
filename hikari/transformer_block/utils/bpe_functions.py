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

def bpe_cycle(byte_corpus, iterations, merge_rules):
    if iterations <= 0:
        return byte_corpus, merge_rules
    
    print("cycle: ", iterations)

    shared_dictionary = {}
    for sublist in byte_corpus:
        count_result = count_pairs(sublist)
        for pair, count in count_result.items():
            if pair in shared_dictionary:
                shared_dictionary[pair] += count
            else:
                shared_dictionary[pair] = count
                

    selected_pair = select_highest_count(shared_dictionary)
    if selected_pair is None:
        return byte_corpus, merge_rules
    
    merge_rules.append(selected_pair)
    print("merges: ", merge_rules)

    new_corpus = []
    for sublist in byte_corpus:
        word = merge(sublist, selected_pair)
        new_corpus.append(word)

    print("result:", new_corpus)

    return bpe_cycle(new_corpus, iterations -1, merge_rules)

def build_vocab(merge_rules):
    byte_vocab = {}
    token_id = 256

    for i in range(256):
        token = bytes([i])
        byte_vocab[token] = i

    for pair in merge_rules:
            token = pair[0] + pair[1]
            if token in byte_vocab:
                continue
            else:
                byte_vocab[token] = token_id
                token_id += 1

    return byte_vocab

def tokenize_bytes(text, merge_rules):
    tokens = [bytes([b]) for b in text.encode("utf-8")]

    for pair in merge_rules:
        tokens = merge(tokens, pair)
        
    return tokens

def encode(map, tokens):
    encoded_text = []

    for token in tokens:
        encoded_text.append(map[token])

    return encoded_text

def decode(inverted_map, token_ids):
    decoded_list = []

    for id in token_ids:
        token = inverted_map[id]
        decoded_list.append(token)

    
    decoded_text = b"".join(decoded_list)
    return decoded_text


"""#LEARNED MERGES
merge_rules = [
    (b'a', b't'), (b'h', b'e'), (b'he', b' '),
    (b'at', b' '), (b'T', b'he '), (b'The ', b'c'), 
    (b'The c', b'at '), (b's', b'at '), (b' ', b't'), 
    (b' t', b'he '), (b'at', b'.'), (b'The cat ', b'sat '), 
    (b'The cat sat ', b'o'), (b'The cat sat o', b'n'),
    (b'The cat sat on', b' the '), (b'The cat sat on the ', b'm'),
    (b'The cat sat on the m', b'at.'), (b'The cat ', b'w'),
    (b'The cat w', b'a'), (b'The cat wa', b's')]"""


merge_rules = [(b'a', b't'), (b'h', b'e'), (b'he', b' '), (b'at', b' '), (b'T', b'he ')]


"""
inverted_map = {
    token_id: token
    for token, token_id in byte_vocab.items()
}
"""


