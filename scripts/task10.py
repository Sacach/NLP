import nltk
from nltk.corpus import nps_chat as nps
from nltk.corpus import stopwords
from scipy import spatial
import numpy as np
from empath import Empath
from itertools import combinations
def main():
    lexicon = Empath()
    library = {}
    for file in nps.fileids():
        posts = nps.xml_posts(file)
        for post in posts:
            key = post.get('class').lower()
            if key not in library:
                library[key] = ""
            library[key] += post.text + " "
        
    top_values = [[0 for i in range(10)],[() for i in range(10)]]
    empath_analysis = {}
    for key in library:
        empath_analysis[key] = lexicon.analyze(library[key], normalize=True)

    for cat1, cat2 in combinations(empath_analysis, 2): 
        if cat1 != "other" and cat2 != "other":
            if sum(list(empath_analysis[cat1].values())) > 0 and sum(list(empath_analysis[cat2].values())) > 0:
                result = 1 - spatial.distance.cosine(list(empath_analysis[cat1].values()), list(empath_analysis[cat2].values()))
                print("Cosine similarity between {} and {} is {:.2f}".format(cat1, cat2, result))
            else:
                print("Null vectors in calculation between {} and {}, result undefined ".format(cat1, cat2))
        if result > min(top_values[0]):
            top_values[0][top_values[0].index(min(top_values[0]))] = result
            top_values[1][top_values[0].index(min(top_values[0]))] = (cat1, cat2)
    print("Highest similarities (not in order)")
    for i in range(10):
        print("Cosine similarity between {} and {} is {:.2f}".format(top_values[1][i][0], top_values[1][i][1], top_values[0][i]))
    """"""
if __name__ == "__main__":
    main()
