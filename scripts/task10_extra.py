import nltk
from nltk.corpus import nps_chat as nps
from nltk.corpus import stopwords
from scipy import spatial
import numpy as np
from empath import Empath
from itertools import combinations
def main():
    lexicon = Empath()
    library = [{},{},{},{},{}]
    ages = ["teens", "20s", "30s", "40s", "adults"]
    for file in nps.fileids():
        if "teens" in file:
            posts = nps.xml_posts(file)
            for post in posts:
                key = post.get('class').lower()
                if key not in library[0]:
                    library[0][key] = ""
                library[0][key] += post.text + " "
        if "20s" in file:
            posts = nps.xml_posts(file)
            for post in posts:
                key = post.get('class').lower()
                if key not in library[1]:
                    library[1][key] = ""
                library[1][key] += post.text + " "
        if "30s" in file:
            posts = nps.xml_posts(file)
            for post in posts:
                key = post.get('class').lower()
                if key not in library[2]:
                    library[2][key] = ""
                library[2][key] += post.text + " "
        if "40s" in file:
            posts = nps.xml_posts(file)
            for post in posts:
                key = post.get('class').lower()
                if key not in library[3]:
                    library[3][key] = ""
                library[3][key] += post.text + " "
        if "adults" in file:
            posts = nps.xml_posts(file)
            for post in posts:
                key = post.get('class').lower()
                if key not in library[4]:
                    library[4][key] = ""
                library[4][key] += post.text + " "
    empath_analysis = [{},{},{},{},{}]
    for i, age_group in enumerate(library):
        for key in age_group:
            empath_analysis[i][key] = lexicon.analyze(age_group[key], normalize=True)
    for book1, book2 in combinations(empath_analysis, 2): 
        print("Comparing {} and {}".format(ages[empath_analysis.index(book1)], ages[empath_analysis.index(book2)]))
        for i in book1:
            if sum(list(book1[i].values())) > 0 and sum(list(book2[i].values())) > 0:
                result = 1 - spatial.distance.cosine(list(book1[i].values()), list(book2[i].values()))
                print("Cosine similarity in dialog act {} is {:.2f}".format(i ,result))
if __name__ == "__main__":
    main()
