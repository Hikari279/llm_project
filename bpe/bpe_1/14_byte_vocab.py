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

merge_rules = [
    (b'l', b'o'),
    (b'lo', b'w'),
    (b'lo', b'\xc3'),
    (b'lo\xc3', b'\xa9'),
    (b'low', b'e')
]


byte_vocab = build_vocab(merge_rules)

print(len(byte_vocab))
print(byte_vocab[b'lo'])
print(byte_vocab[b'low'])
print(byte_vocab[b'lo\xc3\xa9'])