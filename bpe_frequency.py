word = "banana"

symbols = [char for char in word]

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

print(word)
print(frequency)
