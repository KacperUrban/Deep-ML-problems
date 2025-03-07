import numpy as np
from collections import Counter, defaultdict

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
	if not corpus:
		raise ValueError("Corpus is empty!")

	counter_list = [Counter(document) for document in corpus]
	idf_for_queries = defaultdict(int)
	N = len(corpus) + 1
	tf_idf = []

	for q in query:
		df_t = 1
		for counter in counter_list:
			if counter[q]:
				df_t += 1
		idf_for_queries[q] = np.log(N / df_t) + 1

	for counter in counter_list:
		document = []
		for q in query:
			tf = counter[q] / counter.total()
			document.append(round(tf * idf_for_queries[q], 5).item())
		tf_idf.append(document)

	return tf_idf
