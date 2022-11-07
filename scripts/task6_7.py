import nltk
from nltk.corpus import nps_chat as nps
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import re
from pprint import pprint
from matplotlib import pyplot as plt



#Preprocessing functions
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

#Initialize needed variables
stop_words = stopwords.words('english')
stop_words.extend(["action","join"])
accepted_inputs = ["teens","20s","30s","40s","adults"]
forbidden = ["s","m"]

#Calculates topics for a given age group
def get_topics(key):
    #preprocessing pipeline, sourced from: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    sentences = []
    for file in nps.fileids():
        if str(key) in file:
            posts = nps.posts(file)
            for post in posts:
                sentence = ""
                for word in post:
                    sentence += word + " "
                sentences.append(sentence)           
    data = []
    data = [re.sub('JOIN', '', sent) for sent in sentences]
    data = [re.sub('PART', '', sent) for sent in sentences]
    data_words = list(sent_to_words(data))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    nlp = spacy.load("en_core_web_sm")
    data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    for i in reversed(texts):
        for j in reversed(i):
            if j in forbidden:
                i.remove(j)
        if len(i)==0:
            texts.remove(i)
    corpus = [id2word.doc2bow(text) for text in texts]
    #Generate and display topic definitions
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    pprint(lda_model.print_topics(10))
    return lda_model, id2word
    
    #Calculate coherence of a given LDA model
def get_coherence(model, converter):
    topic_words = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(10):
        for word in model.get_topic_terms(i):
            topic_words[i].append(converter[word[0]])
            
    maksimi = 0
    for topic1 in topic_words:
        for word in topic1:
            count = 0
            for topic2 in topic_words:
                if word in topic2:
                    count += 1
            if count > maksimi:
                maksimi = count
                worst = word
    #Coherence as defined by the project specification
    coherence = 1 - (maksimi - 1)/10
    print("Most common word in topic definitions is {} with {} occurances".format(worst, maksimi))
    print("Coherence of the LDA classification is: ", coherence)

#Get coherence between topic definitions of two age groups
def get_inter_group_coherence(model1, model2, converter1, converter2):
    model1_words = []
    model2_words = []
    common_words = []
    
    for i in range(10):
        for word in model1.get_topic_terms(i):
            model1_words.append(converter1[word[0]])
    
    for i in range(10):
        for word in model2.get_topic_terms(i):
            model2_words.append(converter2[word[0]])

    # We propose a definition for inter-group coherence as follows:
    # Calculate amount of common words between the two groups' topic 
    # definitions and subtract proportion of amount of common words 
    # and total words in the smaller definition by word count from 1.
    # Coherence = 1 - common_words / amount_of_words_in_smaller_definition
    # If definitions are identical, coherence = 0 and if definitions are 
    # perfectly unique, coherence = 1. Calculation is symmetric.
    count = 0
    for word in sorted(set(model1_words)):
        if word in model2_words:
            count += 1
            common_words.append(word)
    normalize = min(len(model1_words),len(model2_words))
    inter_group_coherence = 1 - count/normalize
    print("inter-group coherence: ",inter_group_coherence)
    print("common words between topic definitions: ")
    print(sorted(common_words))

#Choose to run in single or multi mode    
def main():    
    mode = input("Choose mode (single corpus coherence (S), inter-group coherence(M))").lower()
    if mode == "s":
        a = input("Give input: (teens, 20s, 30s, 40s, adults)").lower()
        if a not in accepted_inputs:
            print("invalid input!")
            return 0
        #task6
        x, y = get_topics(a)
        #task7a
        get_coherence(x, y)
    
    #task7b
    elif mode == "m":
        print("Select two age-groups to compare (teens, 20s, 30s, 40s, adults)")
        a = input("First group: ").lower()
        b = input("Second group: ").lower()
        if (a not in accepted_inputs) or (b not in accepted_inputs):
            print("One or more inputs invalid!")
            return 0
        x1, y1 = get_topics(a)
        x2, y2 = get_topics(b)
        get_inter_group_coherence(x1,x2,y1,y2)
        
    
        
        
if __name__ == "__main__":
    main()