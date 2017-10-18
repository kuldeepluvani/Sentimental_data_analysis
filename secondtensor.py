import io
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer ## actually removes similar words like running, run, ran. These all are similar words.
## as in sentemental analysis, tens word doesnt matter much.

import numpy as np 
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

## here you may go out of memory....as we are dealing with very big dataset. 
## MemoryError - You are out of RAM.

def create_lexicon(pos,neg):
	lexicon = []
	for fi in [pos,neg]:
		with io.open(fi, 'r', encoding = 'cp437') as f:
			contents = f.readlines()
			#contents = contents.encode('ascii', 'ignore')
			for l in contents[:hm_lines]:
				#l = l.encode('ascii', 'ignore')
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)	

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	print lexicon
	w_counts = Counter(lexicon)
	print 'W count =', len(w_counts)

	l2 = []

	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	print len(l2)
	return l2



#Classify feature set
def sample_handling(sample, lexicon, classification):
	featureset = []

	with io.open(sample, 'r', encoding = 'cp437') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1			

		features = list(features)
		featureset.append([features, classification])

	return featureset


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt',lexicon,[1,0])
	features += sample_handling('neg.txt',lexicon,[0,1])
	random.shuffle(features)

	#what does neural network does for this analysis?
	# does tf.argmax([output]) == tf.argmax([expectations])?

	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
with open('sentiment_set.pickle', 'wb') as f:
	pickle.dump([train_x, train_y, test_x, test_y],f)





