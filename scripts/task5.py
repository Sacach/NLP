import nltk
from nltk.corpus import nps_chat as nps
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from wordcloud import WordCloud
from operator import itemgetter
from matplotlib import pyplot as plt
import re


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
                
       

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]
    
def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    
stop_words = stopwords.words('english')
stop_words.extend(["action","join"])
forbidden = ["s","m","d","i"]
age_group_corpus = [[],[],[],[],[]]

for i,age_group in enumerate(sentences):
    print("Round ", i+1, "out of 5")
    data = []
    data = [re.sub('JOIN', '', sent) for sent in age_group]
    data = [re.sub('PART', '', sent) for sent in age_group]

    print("Cleaning the corpus...")

    data_words = list(sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    print("Removing stop words...")
    data_words_nostops = remove_stopwords(data_words)
    print("Forming bigrams...")
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    print("Loading spacy...")
    nlp = spacy.load("en_core_web_sm")
    print("Lemmatizing...(takes a few seconds)")
    data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    texts = data_lemmatized
    print("Final fixes...")
    for x in reversed(texts):
        for y in reversed(x):
            if y in forbidden:
                x.remove(y)
        if len(x)==0:
            texts.remove(x)
    
    age_group_corpus[i].append(texts)

age_group_list = [{},{},{},{},{}]

for i,age_group in enumerate(age_group_corpus):
    age_group = age_group[0]
    for sent in age_group:
        for word in sent:
           if word not in age_group_list[i]:
                age_group_list[i][word] = 1
           else:
                age_group_list[i][word] += 1 

age_group_length = []
for i in age_group_list:
    count=0
    for word in i:
        count += i[word]
    age_group_length.append(count)

for j,i in enumerate(age_group_list):
    for word in i:
        i[word] /= age_group_length[j]
        
for i,j in enumerate(age_group_list):
    age_group_list[i] = dict(sorted(j.items(), key=itemgetter(1), reverse=True))
    
wl0 = list(age_group_list[0])
wl1 = list(age_group_list[1])
wl2 = list(age_group_list[2])
wl3 = list(age_group_list[3])
wl4 = list(age_group_list[4])

uniques = []
for word in wl0:
    otherwords = set(wl1+wl2+wl3+wl4)
    if word not in otherwords:
        uniques.append((word))
print("Some unique words from group \"teens\"",uniques[0:10])

uniques = []
for word in wl1:
    otherwords = set(wl0+wl2+wl3+wl4)
    if word not in otherwords:
        uniques.append((word))
print("Some unique words from group \"20s\"",uniques[0:10])

uniques = []
for word in wl2:
    otherwords = set(wl1+wl0+wl3+wl4)
    if word not in otherwords:
        uniques.append((word))
print("Some unique words from group \"30s\"",uniques[0:10])

uniques = []
for word in wl3:
    otherwords = set(wl1+wl2+wl0+wl4)
    if word not in otherwords:
        uniques.append((word))
print("Some unique words from group \"40s\"",uniques[0:10])

uniques = []
for word in wl4:
    otherwords = set(wl1+wl2+wl3+wl0)
    if word not in otherwords:
        uniques.append((word))
print("Some unique words from group \"adults\"",uniques[0:10])
   
print("The top 20 words for group \"teens\"")
print(list(age_group_list[0])[0:20])

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(list(age_group_list[0])[0:20]))
                
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

print("The top 20 words for group \"20s\"")
print(list(age_group_list[1])[0:20])

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(list(age_group_list[1])[0:20]))
                
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

print("The top 20 words for group \"30s\"")
print(list(age_group_list[2])[0:20])

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(list(age_group_list[2])[0:20]))
                
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

print("The top 20 words for group \"40s\"")
print(list(age_group_list[3])[0:20])

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(list(age_group_list[3])[0:20]))
                
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

print("The top 20 words for group \"adults\"")
print(list(age_group_list[4])[0:20])

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(list(age_group_list[4])[0:20]))
                
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()








