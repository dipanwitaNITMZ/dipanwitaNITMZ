from rouge import rouge
import nltk.data
import codecs
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import _stop_words
import random
import numpy as np
import random as rn
import numpy as np
import matplotlib.pyplot as plt

#from nsgacrowdingdis import *
from random import randrange, uniform
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import euclidean_distances
from gensim.models import Word2Vec
from pyemd import emd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

#model = Word2Vec.load("data/word_model.mod")

import numpy.matlib
import numpy as np

model = Word2Vec.load("data\word_model.mod")




count =0
document_name ="d04a/"
am ="d04aa"
dm ="FT923-5089"
doc2 = codecs.open('D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep1.txt', "w")
head = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/head.txt")
prep = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep2.txt")

mystopwords = ["one", "on", "also", "next", "ask", "set", "the", "for", "show" , "now", "need", "post" , "said"]



def prepdata():
    ps = PorterStemmer()
    # word_tokenize accepts a string as an input, not a file.
    stop_words = set(stopwords.words('english'))
    data = ' '
    headline = " "
    originallength = []
    # with open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/FT923-5089") as f:
    with open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/" + document_name + dm) as f:
        with open( "D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/head.txt","w") as f1:
            for line in f:
                data = data + line.replace('\n', ' ')
            #print(data)
            firstDelPos = data.find("<TEXT>")  # get the position of delimiter [
            secondDelPos = data.find("</TEXT>")  # get the position of delimiter ]
            extractedString = data[firstDelPos + 6:secondDelPos]  # get the string between two dels
            #print(extractedString)
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            doc2.write('\n'.join(tokenizer.tokenize(extractedString)))

           # print(headline)
            firstDelPos = data.find("<HEADLINE>")  # get the position of delimiter [
            secondDelPos = data.find("</HEADLINE>")  # get the position of delimiter ]
            extractedString = data[firstDelPos + 10:secondDelPos]  # get the string between two dels
            extractedString = extractedString.lower()
            words = extractedString.split(" ")
            stopdata = " "
            for r in words:
                if not r in stop_words:
                    if not r in mystopwords:
                        stopdata = stopdata + " " + r
            stopdata = re.sub(r'\([^)]*\)', '', stopdata)
            stopdata = re.sub('[^A-Za-z0-9\s]+', '', stopdata)
            #print(stopdata)
            f1.write(stopdata)
            f1.close()
            doc2.close()
            file1 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep1.txt")
            appendFile = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep2.txt",'w')
            for line in file1:
                #print(line)
                line = line.lower()
                line = re.sub(r'\([^)]*\)', '', line)
                line = re.sub('[^A-Za-z0-9\s]+', '', line)
                words = line.split(" ")
                m = len(words)
                originallength.append(m)
                stopdata = " "
                for r in words:
                    if not r in stop_words:
                        if not r in mystopwords:
                            stopdata = stopdata + " " + r

                #print(stopdata)
                appendFile.write(stopdata)

            #print(originallength)
            print("original length",len(originallength))
            return originallength



def prepfeature():
   

    print(wmdd)
    return wmdd,count


def create_starting_population(individuals, chromosome_length):
       

        return population


def create_reference_solution(chromosome_length):
  
    return reference


def populationlength(population):
       

            list.append(sumpop)
            # print(list)
        list.sort()

        newpopulation = np.delete(population, indexi, axis=0)
        # print(newpopulation)
        return newpopulation


def ob
        # print(count)
        # print(indexi)

        arrop = numpy.array(indexi)
        arrop = arrop.reshape((-1, 1))
        # print(arrop.shape)
        # print(arrop)
        return arrop



def breed_by_crossover(parent_1, parent_2):
    """
    
    # Return children
    return child_1,child_2


def randomly_mutate_population(population, mutation_probability):
    """
   


def breed_population(population):
  
    return population


def lenghi(cplen):
    
    return sumpop



def objective11(inip, xf):
   
            sums = sums + (cp[cpo] *xf[cpo])

        if leng == 0:
            sums = 0
        sums = float(sums)
        indexi.append(sums)

        # print(initial_population[i])
        # newvalue = np.array(initial_population[i] * np.array(xf))
        # print(newvalue)
        # cp = initial_population[i]
        # obj1 = np.sum(newvalue, axis=1, keepdims=True)
        # print(obj1)

        count = count + 1
        # print(count)
        # print(indexi)
        # print(indexi.shape)
    arrop = numpy.array(indexi)
    arrop = arrop.reshape((-1, 1))
    # print(arrop.shape)
    return arrop




def ovjective22(popobj, wmdsim):
   

        popum = popum + 1
        if leng == 0:
            mm = 0
        #print( popum)
        indexi.append(mm)
    #print(count)
    #print(indexi)

    arrop = numpy.array(indexi)
    arrop = arrop.reshape((-1, 1))
    #print(arrop.shape)
    #print(arrop)
    return arrop



def identify_pareto(scores, population_ids):
    """
   
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]



def mr(cm):
    number_of_ones = int(cm /2)


    return reference


def tmo(npi , newdata):
    
    arr = np.concatenate((objective_score1, objective_score2), axis=1)
    aresum = np.sum((objective1_score, objective2_score), axis=0)


    li =0
    #for em in aresum:
    #    print("em",em)
    sortedvalue = np.sort(aresum, axis=0)[::-1]
    #print("sorted value",sortedvalue[0])
    for eachpop in range(0, cm-1):
        if (((aresum[eachpop][0]) == sortedvalue[0])):
            #print("found match",eachpop)
            x = tmdata[eachpop]

    #print(x)
    return x, sortedvalue[0]

    #print(x)



def rev(a,svc,smc):
   
    return a





def childgeneration(nm, cpbest, cpbestvalues, gbest, gbestvalue):
   

    #print(x)
    return a, cpbest,cpbestvalues



m = prepdata()
prepfeature()
fc = obj1()
wmdd,count = obj2()



xf = fc

wmdsim = wmdd

k = 100
chromosome_length = count

fs = [1, 2, 3, 5, 4, 6]
individuals = chromosome_length * 5
population_size = (chromosome_length * 5)
maximum_generation = 2
best_score_progress = []
summarylimit = 110

# Tracks progress


# Create reference solution
# (this is used just to illustrate GAs)
reference = create_reference_solution(chromosome_length)
#print("ref",reference)

# Create starting population
population = create_starting_population(population_size, chromosome_length)
#print("initial population", population)

initial_population = populationlength(population)
cm = 0
for we in initial_population:
    cm =cm + 1

#print("no of population",cm)




currentpbest = initial_population
aresum = np.sum((objective1_score, objective2_score), axis=0)
currentpbestvalues = aresum
newdata = []
print("current pbest population", currentpbest)
print("current pbest population objective score", currentpbestvalues)
plt.scatter(aresum,aresum)
plt.show()

#print(aresum)
sortedvalue = np.sort(aresum,axis=0)[::-1]


gbest=newdata[0]
gbestvalue = sortedvalue[0]
print("gbest =",newdata[0])
population1 = initial_population
npopulation = population1
for generation in range(maximum_generation):
    print(" generation ", generation)
    #print("cm =", cm)
    for i in range(0, cm - 1):

        nm = population1[i]
        newi,pbesti,currentpbestvalue = childgeneration(nm,currentpbest[i],currentpbestvalues[i],gbest,gbestvalue,)
        x,newval = tmo(nm, newdata)
        population1[i,:]=newi
        currentpbestvalues[i,:]=currentpbestvalue
        currentpbest[i, :]= pbesti

        print("current pbest",newval)
        print("new tm updated data",population1[i])
        if(gbestvalue<currentpbestvalue):
            gbest = pbesti
            gbestvalue=currentpbestvalue



    for i in gbest:
        print("newdata",i)
m = []
print("best pop", gbest)
for i in gbest:
    m.append(i)

print(m)
prepp = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/prep1.txt","r")
op = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/op.txt","w")
x = m

iv = 0
print("optimal cat",m[iv])
output = " "
for line in prepp:
    if x[iv].__eq__(1.0):
        output = output + line.replace('\n', '.')
    iv = iv + 1

print("obtained output",output)
op.write(output.lower())

op.close()


data = []
s = " "
i = 0
extractedString  =" "
f2 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/orisum.txt",'w')
with open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/original.summaries/"+ am +"/perdocs") as f1:
#with open("D:/Dipanwita docs/cat swarm/summaries/summaries/" +am+"/perdocs") as f1:
    for line in f1:
        i = i + 1
        data.append(line)




for j in range (0, i-1):
    if data[j].__contains__(dm):
        for k in range (j+3,i-1):
            extractedString = extractedString + data[k]
            if data[k].__contains__("</SUM>"):
                break
        break

print("Actual summary",extractedString)

extractedString = extractedString.replace("\n","")
extractedString =extractedString.lower()
f2.write(extractedString)

plt.scatter(aresum,aresum)
plt.show()


f2.close()

f1 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/op.txt")
f2 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/orisum.txt")
#f4 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/archive.csv", "a")
#f3 = open("D:/Dipanwita docs/cat swarm/DUC2001_Summarization_Documents/DUC2001_Summarization_Documents/data/test/docs/d04a/archive.txt", "a")
f4 = open("D:/Dipanwita docs/phd/output/csowithdepsocsoGA.csv", "a")
f3 = open("D:/Dipanwita docs/phd/output/csowithdepsocsoga.txt", "a")

#f2 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/actualsumm.txt","r")
#f1 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/obtainedsumm.txt","r")
#f4 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/output.csv", "a")
#f3 = open("D:/Dipanwita docs/scientific document/scisummnet_release1.1__20190413/top1000_complete/1aoutput/outputtxt.txt","a")
#from gasumm import *
#from cd1fea import document_name



arr1 = ""
for line2 in f2:
    arr1 = line2


arr2 = " "
for line in f1:
    arr2 = line



li = "A00-1031"

#scores = Rouge.get_scores(arr1,arr)
scores =rouge.rouge_n_sentence_level(arr2, arr1, 1)
print("rough 1: " , scores)
li = li + ","+str(scores)+","
scores =rouge.rouge_n_sentence_level(arr2, arr1, 2)
print("rough 2: ", scores)
li = li + str(scores)
li = li.replace("=","")
li = li.replace("(","")
li = li.replace(")","")

li = li.replace("RougeScorerecall","")
li = li.replace("f1_measure","")
li = li.replace("precision","")
m = li
li =   li + "\t ORIGINAL TEXT" + arr1 +"\t\t  SYSTEM GENERATED OUTPUT" +arr2

print(li)
f3.write(li)
f3.write("\n")
f4.write(m)
f4.write("\n")








































