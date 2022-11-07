import nltk
from nltk.corpus import nps_chat as nps
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import uncertainties.unumpy as unp
import uncertainties as unc
from scipy.optimize import curve_fit
from scipy import stats
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np
import re

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

#Linear fit and confidence calculus functions    
def f(x, a, b):
    return a * x + b
    
def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

#Initialize needed variables
stop_words = stopwords.words('english')
stop_words.extend(["action","join"])
forbidden = ["s","m","suser","adultsuser","teensuser"]
posts = nps.xml_posts()
book = {}

#Gather all sentences in each dialogue act into a dictionary with the sentence's dialogue act tag
for post in posts:
    key = post.get('class').lower()
    if key not in book:
        book[key]={}
        book[key][post.text]=1
    else:
        if post.text not in book[key]:
            book[key][post.text]=1
        else:
            book[key][post.text]+=1
#preprocessing pipeline, sourced from: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
clean_dict = {}
for i in book:
    sentences = []
    print("Cleaning dialogue act", i)
    for sentence in book[i]:
        for repeat in range(book[i][sentence]):
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
    data_lemmatized = lemmatization(nlp, data_words_bigrams )
    texts = data_lemmatized
    for k in reversed(texts):
        for j in reversed(k):
            if j in forbidden:
                k.remove(j)
        if len(k)==0:
            texts.remove(k)
            
    clean_dict[i] = texts

#Count word occurances in the corpus for each dilogue act tag
frequency_dict = {}
dialogue_length_dict = {}
for i in clean_dict:
    totalcount = 0
    frequency_dict[i]={}
    for sentence in clean_dict[i]:
        for word in sentence:
            if word not in frequency_dict[i]:
                frequency_dict[i][word] = 1
            else:
                frequency_dict[i][word] += 1
            totalcount += 1
    
    dialogue_length_dict[i]=totalcount

#Convert occurances to frequencies
for i in frequency_dict:
    for word in frequency_dict[i]:
        frequency_dict[i][word] /= dialogue_length_dict[i]
#Reorder data so that highest frequency takes the 0-th index lists for ranks and frequencies
for i in frequency_dict:
    frequency_dict[i] = dict(sorted(frequency_dict[i].items(), key=itemgetter(1),reverse=True))
    rank = range(len(frequency_dict[i]))
    frequencylist = []
    for word in frequency_dict[i]:
        frequencylist.append(frequency_dict[i][word])
    #Plot the frequency vs. rank log-log plot for dialogue act categories (categories with less than 101 unique words ignored by default)
    if len(frequency_dict[i])>100:
        plt.plot(rank, frequencylist, label = "{}".format(i))
        plt.legend()

print("Choose one of the shown dialogue act tags for further analysis")
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Word Frequency')
plt.title('Log-log plot of word frequency vs rank by dialogue act tag')
plt.show()

#Choose a dialogue act tag to analyze further and scale the data (data = ln(data))
valid_keys = ["Statement","Emotion","System","Greet","Accept","Reject","whQuestion","Continuer","ynQuestion","Bye","Emphasis"]
frequencies = []
key = input("Type the chosen tag (Statement,Emotion,System,Greet,Accept,Reject,whQuestion,Continuer,ynQuestion,Bye,Emphasis): ").lower()
if key not in valid_keys:
    print("Invalid key")
for i in frequency_dict[key]:
    frequencies.append(frequency_dict[key][i])

ranks = [i for i in range(len(frequencies))]

for j,i in enumerate(ranks):
    ranks[j] = np.log(i+1)
for j,i in enumerate(frequencies):
    frequencies[j] = np.log(i)

#Calculate linear fit and confidence interval for scaled data, sourced from: https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
x = np.array(ranks)
y = np.array(frequencies)
n = len(y)
popt, pcov = curve_fit(f, x, y)
a = popt[0]
b = popt[1]
print('Optimal Values')
print('Slope: ' + str(a))
print('Constant: ' + str(b))
r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
a,b = unc.correlated_values(popt, pcov)

px = np.linspace(0, max(ranks)+0.4, 1000)
py = a*px+b
nom = unp.nominal_values(py)
std = unp.std_devs(py)
lpb, upb = predband(px, x, y, popt, f, conf=0.90)

#Plot scaled data with linear fit and confidence bounds in linear scale plot
plt.plot(px, nom, c='black', label='y=a x + b')
plt.plot(px, lpb, 'k--',label='90% Confidence interval')
plt.plot(px, upb, 'k--')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')
plt.scatter(ranks, frequencies, label = "raw data")
plt.xlabel('(natural log of) Word Rank')
plt.ylabel('(natural log of) Word Frequency')
plt.title('Word frequency vs word rank in dialogue act "{}"'.format(key))
plt.ylim(min(frequencies)-0.2,max(frequencies)+0.15)
plt.xlim(-0.4, max(ranks)+0.4)
plt.legend()
plt.show()

#Estimate the proportion of data points outside the condidence bound
out_count = 0
for i in range(len(frequencies)):
    x_sample = ranks[i]
    y_sample = frequencies[i]
    y_compare_low = (lpb[1]-lpb[0])/(px[1]-px[0]) * x_sample + lpb[0]
    y_compare_high = (upb[1]-upb[0])/(px[1]-px[0]) * x_sample + upb[0]
    if y_sample < y_compare_low or y_sample > y_compare_high:
        out_count += 1

out_proportion = out_count/len(ranks) * 100
print("Approximately {:.2f}% of samples are outside the confidence interval".format(out_proportion))