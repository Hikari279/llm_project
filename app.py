import tiktoken

# Choose a specific encoding
enc = tiktoken.get_encoding("o200k_base")

text = "I'm actually still reading the notes. it might take me 5-10 mins to finish. you can start the next steps yourself if u like"

# Convert text into token IDs
ids = enc.encode(text)

# Convert IDs into readable token pieces
pieces = [enc.decode_single_token_bytes(i) for i in ids]

print("Text:", text)
print("Token count:", len(ids))
print("Token IDs:", ids)
print("Token pieces:", pieces)

# Convert the IDs back into text
print("Decoded:", enc.decode(ids))