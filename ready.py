from nltk import PorterStemmer, word_tokenize

in_txt = "Let's learn natural language processing with python programming"

Toekn = word_tokenize(in_txt)

print Toekn


stm = PorterStemmer()

stemmed_word = [stm.stem(token) for token in Toekn]

print stemmed_word