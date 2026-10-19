#task 1

"""text = "hello"
encoded = text.encode("utf-8")

print(encoded)
print(list(encoded))

text = "café 😊"
encoded = text.encode("utf-8")

print(encoded)
print(list(encoded))
print(len(text))
print(len(encoded))"""

#task2

"""text = "café 😊"

byte_tokens = list(text.encode("utf-8"))

print(byte_tokens)
print(type(byte_tokens[0]))

byte_tokens = [bytes([b]) for b in text.encode("utf-8")]

print(byte_tokens)"""

#task3
a = bytes([195])
b = bytes([169])

print(a + b)
print((a + b).decode("utf-8"))