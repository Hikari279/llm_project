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


#task 1

"""text = "café 😊"
byte_tokens = [bytes([b]) for b in text.encode("utf-8")]

print(byte_tokens)

pair = (byte_tokens[0], byte_tokens[1])

result = merge(byte_tokens, pair)
print(result) """

#task 2

text = "café 😊"
byte_tokens = [bytes([b]) for b in text.encode("utf-8")]

print(byte_tokens)

pair = (byte_tokens[3], byte_tokens[4])

result = merge(byte_tokens, pair)

print(result)
print(result[3])
print(result[3].decode("utf-8"))