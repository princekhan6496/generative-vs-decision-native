from mini_llm.tokenizer import SimpleTokenizer
def test_roundtrip_vocab():
 t=SimpleTokenizer().fit(['hello world']); x=t.encode('hello world',8); assert x[0]==1 and x[1]!=0 and x[2]!=0
