import nltk
from nltk.corpus import nps_chat as nps
from operator import itemgetter
from matplotlib import pyplot as plt
from empath import Empath
import numpy as np
from scipy import spatial
from itertools import combinations
#Choose to run in single or multi mode    
def main():
    lexicon = Empath()
    ages = ["teens", "20s", "30s", "40s", "adults"]
    """
    print("Choose two age groups for empath category comparison.")
    print("(teens, 20s, 30s, 40s, adults)")
    group1 = input("Give a name of an age group: ").lower()
    if group1 in ages:
        group2 = input("Give a name of an age group 2: ").lower()
    else:
        print("not avalid input")
        return 0
    if group2 not in ages:
        return 0
    index1 = ages.index(group1)   
    index2 = ages.index(group2)
    """
    #print(index1, index2)
    totalcount = 0
    sentences = [[],[],[],[],[]]
    print("Forming sentences...")
    for file in nps.fileids():
        if "teens" in file:
            for post in nps.posts(file):
                sentence = ""
                for word in post:
                    sentence += word + " "
                sentences[0].append(sentence)
        if "20s" in file:
            for post in nps.posts(file):
                sentence = ""
                for word in post:
                    sentence += word + " "
                sentences[1].append(sentence)
        if "30s" in file:
            for post in nps.posts(file):
                sentence = ""
                for word in post:
                    sentence += word + " "
                sentences[2].append(sentence)
        if "40s" in file:
            for post in nps.posts(file):
                sentence = ""
                for word in post:
                    sentence += word + " "
                sentences[3].append(sentence)
        if "adults" in file:
            for post in nps.posts(file):
                sentence = ""
                for word in post:
                    sentence += word + " "
                sentences[4].append(sentence)
    empath_analysis = []

    for posts in sentences:
        age_group = " ".join(posts)
        #print(age_group)
        empath_analysis.append(lexicon.analyze(age_group, normalize=True))
    for pair in combinations(empath_analysis, 2):    
        #print(empath_analysis)
        list1=[]
        list2=[]
        for i in pair[0].values():
            list1.append(i)
        for i in pair[1].values():
            list2.append(i)
        result = 1 - spatial.distance.cosine(list1, list2)
        print("Cosine similarity between {} and {} is {:.2f}".format(ages[empath_analysis.index(pair[0])], ages[empath_analysis.index(pair[1])], result))

    """
    age_group1 = " ".join(sentences[index1])
    age_group2 = " ".join(sentences[index2])
    empath_analysis1 = lexicon.analyze(age_group1, normalize=True)
    empath_analysis2 = lexicon.analyze(age_group2, normalize=True)
    """

    

    
if __name__ == "__main__":
    main()
