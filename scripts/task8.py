import nltk
from nltk.corpus import nps_chat as nps
from operator import itemgetter
from matplotlib import pyplot as plt
from empath import Empath
import numpy as np
#Choose to run in single or multi mode    
def main():
    lexicon = Empath()
    #print(sorted(lexicon.analyze("hit yellow cat car", normalize=True).items(), key=itemgetter(1), reverse=True))
    ages = ["teens", "20s", "30s", "40s", "adults"]
    
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
    
    for j, posts in enumerate(sentences):
        age_group = " ".join(posts)
        #print(age_group)
        empath_analysis = sorted(lexicon.analyze(age_group, normalize=True).items(), key=itemgetter(1), reverse=True)
        
        print(ages[j])
        objects = []
        percentage = []
        for i in range(10):
            
            print("{}: {:.2f}%".format(empath_analysis[i][0], empath_analysis[i][1]*100))
            objects.append(empath_analysis[i][0])
            percentage.append(empath_analysis[i][1]*100)
        y_pos = np.arange(len(objects))
        objects = tuple(objects)
        plt.bar(y_pos, percentage, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Percentage')
        plt.title('10 most common categories in age-group {}'.format(ages[j]))
        plt.show()
        print("\n")


    """"""
if __name__ == "__main__":
    main()
