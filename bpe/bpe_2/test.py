

from utils.bpe_utils import text_to_byte, text_to_byte_token, get_pair, merge_pair

# texts = ["cat", "é", "😀"]

# for text in texts:
#     BYTES = []
#     encoded = text_to_byte(text)
#     print(encoded)
#     for token in encoded:
#         BYTES.append(text_to_byte_token(token))
#         print(token, type(token))
#         print(BYTES, type(BYTES[-1]))

#     print(get_pair(BYTES))

# tokens = [bytes([195]), bytes([169])]
# pair = (tokens[0], tokens[1])

# merged = merge_pair(tokens, pair)

# print(merged)
# byte_list = list(merged[0])
# print(byte_list)

# new_token = bytes(byte_list)
# print(new_token)

merge_rules = [
    (b'a', b'b'),
    (b'ab', b'c')
]

all_rules = []


for pair in merge_rules:

    all_byte_list = []
    for byte in pair:
        byte_list = list(byte)
        all_byte_list.append(byte_list)

    all_rules.append(all_byte_list)


# print(all_rules)

# restored_rules = []

# for all_byte_list in all_rules:
#     restored_pair = []

#     for byte_list in all_byte_list:
#         byte = bytes(byte_list)
#         restored_pair.append(byte)

#     restored_pair = tuple(restored_pair)
#     restored_rules.append(restored_pair)

# print(restored_rules)

    