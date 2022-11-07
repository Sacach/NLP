import nltk
from nltk.corpus import nps_chat as nps
import uncertainties.unumpy as unp
import uncertainties as unc
import scipy.stats as stats
from scipy.optimize import curve_fit
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np


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
newposts = []
pos_tag_dict = {}
label_list = ["","","",""]

#Minor preprocesing (lower case all letters in words)
for file in nps.fileids():
    chatroom = nps.tagged_posts(file)
    for i,sentence in enumerate(chatroom):
        newposts.append([])
        for word in sentence:
            word = list(word)
            word[0] = word[0].lower()
            word = tuple(word)
            newposts[i].append(word)

#Count word occurances in the corpus for each part of speech tag 
for sentence in newposts:
    for i in sentence:
        if i[1] not in pos_tag_dict:
            pos_tag_dict[i[1]] = {}
            pos_tag_dict[i[1]][i[0]] = 1
        else:
            if i[0] not in pos_tag_dict[i[1]]:
                pos_tag_dict[i[1]][i[0]] = 1
            else:
                pos_tag_dict[i[1]][i[0]] += 1

#Calculate total number of word occurances within each pos tag group
pos_tag_lengths = []
for i in pos_tag_dict:
    count = 0
    for word in pos_tag_dict[i]:
        count += pos_tag_dict[i][word]
    pos_tag_lengths.append(count)

#Convert word occurances to frequencies (seperately within each pos tag group)                    
for j,i in enumerate(pos_tag_dict):
    for word in pos_tag_dict[i]:
        pos_tag_dict[i][word] /= pos_tag_lengths[j]

#Reorder data so that highest frequency takes the 0-th index lists for ranks and frequencies
for i in pos_tag_dict:
    pos_tag_dict[i] = dict(sorted(pos_tag_dict[i].items(), key=itemgetter(1),reverse=True))
    rank = range(len(pos_tag_dict[i]))
    frequencylist = []
    for word in pos_tag_dict[i]:
        frequencylist.append(pos_tag_dict[i][word])
    #Plot the frequency vs. rank log-log plot for pos tag categories (categories with less than 101 unique words ignored by default)
    if len(pos_tag_dict[i])>300:
        plt.plot(rank, frequencylist, label = "pos_tag {}".format(i))
        plt.legend()

print("Choose one of the shown pos-tags for further analysis")
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Word Rank')
plt.ylabel('Word Frequency')
plt.title('Log-log plot of word frequency vs rank by PoS-tag')
plt.show()

#Choose a pos tag to analyze further and scale the data (data = ln(data))
valid_keys = ["RB","VBD","JJ","NN","NNP","UH","VB","CD","VBP","VBG","NNS","VBZ","VBN"]
frequencies = []
key = input("Type the chosen tag (RB,VBD,JJ,NN,NNP,UH,VB,CD,VBP,VBG,NNS,VBZ,VBN): ").upper()

if key not in valid_keys:
    print("Invalid key")

for i in pos_tag_dict[key]:
    frequencies.append(pos_tag_dict[key][i])

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

px = np.linspace(0, max(ranks)+0.4, 100)
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
plt.title('Word frequency vs word rank')
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