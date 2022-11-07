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
forbidden = ["s","m","d","i"]
worddict = {}
totalcount = 0
sentences = []

#preprocessing pipeline, sourced from: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
print("Forming sentences...")
for file in nps.fileids():
    for post in nps.posts(file):
        sentence = ""
        for word in post:
            sentence += word + " "
        sentences.append(sentence)
print("Cleaning the corpus...")
data = []
data = [re.sub('JOIN', '', sent) for sent in sentences]
data = [re.sub('PART', '', sent) for sent in sentences]

data_words = list(sent_to_words(data))

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
print("Removing stop words...")
data_words_nostops = remove_stopwords(data_words)
print("Forming bigrams...")
data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
print("Lemmatizing...(takes several seconds)")
nlp = spacy.load("en_core_web_sm")
data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
texts = data_lemmatized
print("Final fixes...")
for i in reversed(texts):
    for j in reversed(i):
        if j in forbidden:
            i.remove(j)
    if len(i)==0:
        texts.remove(i)

#Count word occurances in the corpus
for sent in texts:
    for word in sent:
        totalcount += 1
        if word not in worddict:
            worddict[word] = 1
        else:
            worddict[word] += 1

#Convert occurances to frequencies
for i in worddict:
    worddict[i] /= totalcount


#Sort the words (most common word in index 0)
sorted_worddict = dict(sorted(worddict.items(), key=itemgetter(1),reverse=True))

#Create lists for ranks and frequencies
ranks = list(range(len(sorted_worddict)))
frequencylist = []
for i in sorted_worddict:
    frequencylist.append(sorted_worddict[i])

#Plot words in rank vs. frecuency log-log plot
plt.scatter(ranks, frequencylist, label='raw data')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1,4000)
plt.xlabel('Word Rank')
plt.ylabel('Word Frequency')
plt.title('Log-log plot of actual word rank vs word frequency')
plt.legend()
plt.show()

#Scale data (data = ln(data))
for j,i in enumerate(ranks):
    ranks[j] = np.log(i+1)
for j,i in enumerate(frequencylist):
    frequencylist[j] = np.log(i)

#Calculate linear fit and confidence interval for scaled data, sourced from: https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics 
x = np.array(ranks)
y = np.array(frequencylist)
n = len(y)
popt, pcov = curve_fit(f, x, y)
a = popt[0]
b = popt[1]
print('Optimal Values')
print('Slope: ' + str(a))
print('Constant: ' + str(b))
r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
a,b = unc.correlated_values(popt, pcov)
px = np.linspace(0, 8.29, 100)
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
plt.scatter(ranks, frequencylist, label = "raw data")
plt.xlabel('(natural log of) Word Rank')
plt.ylabel('(natural log of) Word Frequency')
plt.title('Word frequency vs word rank')
plt.ylim(-9.39,-4.1)
plt.xlim(0,8.3)
plt.legend()
plt.show()

#Estimate the proportion of data points outside the condidence bound
out_count = 0
for i in range(len(frequencylist)):
    x_sample = ranks[i]
    y_sample = frequencylist[i]
    y_compare_low = (lpb[1]-lpb[0])/px[1] * x_sample + lpb[0]
    y_compare_high = (upb[1]-upb[0])/px[1] * x_sample + upb[0]
    if y_sample < y_compare_low or y_sample > y_compare_high:
        out_count += 1

        
out_proportion = out_count/len(ranks) * 100
print("Approximately {:.2f}% of samples are outside the confidence interval".format(out_proportion))