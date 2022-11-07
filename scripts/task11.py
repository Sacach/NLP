import nltk
from nltk.corpus import nps_chat as nps
from nltk.corpus import stopwords
from scipy import spatial
import numpy as np
from empath import Empath
from itertools import combinations
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
def main():

    library = {}
    for file in nps.fileids():
        posts = nps.xml_posts(file)
        for post in posts:
            key = post.get('class').lower()
            if key not in library:
                library[key] = ""
            library[key] += post.text + " "
    data = []
    for key in library:
        for i in sent_tokenize(library[key]):
            temp = []
         
            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())
         
            data.append(temp)
    model1 = gensim.models.Word2Vec(data, min_count = 1,
                              vector_size = 100, window = 5)  
    top_values = [[0 for i in range(10)],[() for i in range(10)]]
    for i, j in combinations(set(nps.words()), 2):

        result = model1.wv.similarity(i, j)
        if result > min(top_values[0]):
            top_values[0][top_values[0].index(min(top_values[0]))] = result
            top_values[1][top_values[0].index(min(top_values[0]))] = (i, j)

    print("Highest similarities (not in order)")
    for i in range(10):
        print("Cosine similarity between {} and {} is {:.2f}".format(top_values[1][i][0], top_values[1][i][1], top_values[0][i]))
    """"""
if __name__ == "__main__":
    main()