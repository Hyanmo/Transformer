import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")

text = "Tabi 是一只斑点狗"
pieces = sp.encode(text, out_type=str)

print(pieces)