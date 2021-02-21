# author: Mohammadhossein Naderi
import os
import glob
import string
import collections
import math
import numpy as np
import heapq as hq
from scipy import spatial
from tabulate import tabulate
from operator import itemgetter


def stemming_rules(term):
    l = len(term)

    if term.startswith("‌"):  # nim fasele
        return term[1:l]

    if term.endswith("‌"):  # nim fasele
        return term[0:l - 1]

    if term.endswith('‌ها') and l > 4:  # nim fasele
        return term[0:l - 3]

    if term.endswith('ات') and l > 4:
        return term[0:l - 2]

    if (term.endswith('ه‌ام') or term.endswith('ه‌ای')) and l > 5:  # nim fasele
        return term[0:l - 3]
    if (term.endswith('ه‌است') or term.endswith('ه‌ایم') or term.endswith('ه‌اید') or term.endswith(
            'ه‌اند')) and l > 6:  # nim fasele
        return term[0:l - 4]

    if term.startswith('می‌'):  # nim fasele
        if (term.endswith('یم') or term.endswith('ید') or term.endswith('ند')) and l > 6:
            return term[3:l - 2]
        if (term.endswith('م') or term.endswith('ی') or term.endswith('د')) and l > 5:
            return term[3:l - 1]
        return term[3:l]

    if term.startswith('می'):
        if (term.endswith('یم') or term.endswith('ید') or term.endswith('ند')) and l > 5:
            return term[2:l - 2]
        if (term.endswith('م') or term.endswith('ی') or term.endswith('د')) and l > 4:
            return term[2:l - 1]
        return term

    if term.startswith('نمی‌'):  # nim fasele
        if (term.endswith('یم') or term.endswith('ید') or term.endswith('ند')) and l > 7:
            return term[4:l - 2]
        if (term.endswith('م') or term.endswith('ی') or term.endswith('د')) and l > 6:
            return term[4:l - 1]
        return term[4:l]

    if term.startswith('نمی'):
        if (term.endswith('یم') or term.endswith('ید') or term.endswith('ند')) and l > 6:
            return term[3:l - 2]
        if (term.endswith('م') or term.endswith('ی') or term.endswith('د')) and l > 5:
            return term[3:l - 1]
        return term

    if term.endswith('‌تر') and l > 4:  # nim fasele
        return term[0:l - 3]
    if term.endswith('‌ترین') and l > 6:  # nim fasele
        return term[0:l - 5]
    if term.endswith('ترین') and l > 5:
        return term[0:l - 4]
    if term.endswith('تر') and l > 3:
        return term[0:l - 2]

    if (term.endswith('ای') or term.endswith('وی') or term.endswith('یی')) and term != 'برای' and l > 3:
        return term[0:l - 1]

    if (term.endswith('دن') or term.endswith('تن')) and (l > 3 or term == 'شدن'):
        return term[0:l - 1]

    if term.endswith('گی') and l >= 4:
        return term[0:l - 2] + 'ه'
    if term.endswith('گان') and l >= 5:
        return term[0:l - 3] + 'ه'
    if (term.endswith('ایان') or term.endswith('ویان')) and l > 5:
        return term[0:l - 3]
    if term.endswith("یان") and l > 5:
        return term[0:l - 2]

    if term.endswith("انه") and l >= 5:
        return term[0:l - 3]

    if term.endswith("نده") and l >= 5:
        return term[0:l - 3]

    if term.endswith("بان") and l > 4:
        return term[0:l - 3]

    if term.endswith("گاه") and l > 4:
        return term[0:l - 3]
    if term.endswith("مند") and l > 4:
        return term[0:l - 3]

    if term.endswith("‌آسا") and l > 5:  # nim fasele
        return term[0:l - 4]
    if term.endswith("ستان") and l > 5:
        return term[0:l - 4]

    return term


def repetitive_stemming(term):
    stemmed_term = stemming_rules(term)
    return term if stemmed_term == term else repetitive_stemming(stemmed_term)


data_path = 'wiki\\'
clusters = ['فیزیک', 'ریاضیات', 'سلامت', 'تاریخ', 'فناوری']
data_pathes = []
for cluster in clusters:
    data_pathes.append(data_path + cluster)



f = {}
inverted_index = {}
champion_lists = {}

for i in range(len(data_pathes)):
    f[i] = []
    for filename in glob.iglob(data_pathes[i] + '**/*.txt', recursive=True):
        f[i].append(os.path.abspath(filename))


term_frequency = {}
for i in range(len(clusters)):
    term_frequency[i] = {}
    for filepath in f[i]:
        file = open(filepath, 'r', encoding='utf-8')
        lines = file.readlines()
        for line in lines:
            exclude = set(string.punctuation).union(
                ['،', '؟', '!', '«', '»', '؛', '١', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۰']).union(string.digits)
            line = ''.join(ch for ch in line if ch not in exclude)
            terms = line.split()
            for term in terms:
                term = repetitive_stemming(term)
                if term in term_frequency[i]:
                    term_frequency[i][term] = term_frequency[i][term] + 1
                else:
                    term_frequency[i][term] = 1

res = {}
stop = {}

for i in range(len(clusters)):
    res[i] = dict(sorted(term_frequency[i].items(), key=itemgetter(1), reverse=True)[:10])
    stop[i] = list(res[i].keys())

for i in range(len(clusters)):
    inverted_index[i] = {}
    for filepath in f[i]:
        file = open(filepath, 'r', encoding='utf-8')
        lines = file.readlines()
        pos = 0
        for line in lines:
            exclude = set(string.punctuation).union(
                ['،', '؟', '!', '«', '»', '؛', '١', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۰']
                                                    ).union(string.digits)
            line = ''.join(ch for ch in line if ch not in exclude)
            terms = line.split()
            for term in terms:
                term = repetitive_stemming(term)
                if term in stop[i]:
                    continue
                pos = pos + 1
                if term not in inverted_index[i]:
                    tmp = {}
                    tmp[filepath] = []
                    tmp[filepath].append(pos)
                    inverted_index[i][term] = tmp
                else:
                    if filepath in inverted_index[i][term]:
                        inverted_index[i][term][filepath].append(pos)
                    else:
                        inverted_index[i][term][filepath] = []
                        inverted_index[i][term][filepath].append(pos)

od = {}
for i in range(len(clusters)):
    od[i] = collections.OrderedDict(sorted(inverted_index[i].items()))

for i in range(len(clusters)):
    for term in od[i]:
        od[i][term] = collections.OrderedDict(sorted(od[i][term].items()))

tfidf = {}
tfs = {}
n = {}
for i in range(len(clusters)):
    n[i] = len(f[i])

def calculate_tfidf():
    for i in range(len(clusters)):
        tfidf[i] = {}
        for filepath in f[i]:
            tmp = []
            for term in od[i]:
                if filepath in od[i][term]:
                    df = len(od[i][term])
                    tf = len(od[i][term][filepath])
                    w = math.log10(1 + tf) * math.log10(n[i] / df)
                    tmp.append(w)
                else:
                    tmp.append(0)
            tfidf[i][filepath] = tmp

def calculate_tfs():
    for i in range(len(clusters)):
        tfs[i] = {}
        for term in od[i]:
            tmp = []
            for filepath in f[i]:
                if filepath in od[i][term]:
                    tmp.append(len(od[i][term][filepath]))
                else:
                    tmp.append(0)
            tfs[i][term] = tmp

def calculate_champion_lists():
    for i in range(len(clusters)):
        champion_lists[i] = {}
        for term in od[i]:
            r = 55
            docs = {}
            j = 0
            for tf in tfs[i][term]:
                docs[f[i][j]] = tf
                j = j + 1
            top_list = []
            for k in range(r):
                max = -1
                max_doc = None
                for h in docs.keys():
                    if docs[h] > max:
                        max = docs[h]
                        max_doc = h
                if max > 0:
                    top_list.append(max_doc)
                    del docs[max_doc]
            champion_lists[i][term] = top_list


centroids = {}

def calculate_centroids():
    for i in range(len(clusters)):
        l = len(tfidf[i][f[i][0]])
        avg = np.zeros(l)
        for filepath in f[i]:
            doc_vector = np.array(tfidf[i][filepath])
            avg = avg + doc_vector
        avg = avg / n[i]
        centroids[i] = avg

calculate_tfidf()
calculate_tfs()
calculate_champion_lists()
calculate_centroids()

#print(champion_lists[])

def test():
    term = input()
    flag = False
    for h in range(len(clusters)):
        if term in od[h]:
            flag = True
            result = od[h][term]
            j = 0
            print(term)
            print("│")
            for doc in result:
                i = 0
                print("│_____" + doc[doc.rfind("\\") + 1:len(doc)])
                for position in od[h][term][doc]:
                    if (j < len(result) - 1):
                        print("│     ", end="")
                    else:
                        print("      ", end="")
                    print("│_____" + str(position))
                    i = i + 1
                if (j < len(result) - 1):
                    print("│")
                j = j + 1
    if not flag:
        print("Not Found!")

def search_query(k):
    query = input()
    tmp = query.split()
    terms = {}
    for i in range(len(clusters)):
        terms[i] = []
        for term in tmp:
            stemmed = repetitive_stemming(term)
            if stemmed not in stop[i]:
                terms[i].append(stemmed)

    qvector = {}
    for i in range(len(clusters)):
        qvector[i] = []
        for term in od[i]:
            if term in terms[i]:
                df = len(od[i][term])
                tf = terms[i].count(term)
                w = math.log10(1 + tf) * math.log10(n[i] / df)
                qvector[i].append(w)
            else:
                qvector[i].append(0)

    max_similarity = -1
    cluster_index = -1
    for i in range(len(clusters)):
        vec = np.array(qvector[i])
        if np.dot(vec, centroids[i]) != 0:
            cosine = (np.dot(vec, centroids[i])) / (np.sqrt(np.sum(vec*vec))*np.sqrt(np.sum(centroids[i]*centroids[i])))
            if cosine > max_similarity:
                max_similarity = cosine
                cluster_index = i

    if cluster_index == -1:
        print("No Result")
        return

    print(clusters[cluster_index])
    intersect = []
    for filepath in f[cluster_index]:
        intersect.append(filepath)

    for term in terms[cluster_index]:
        if term in champion_lists[cluster_index]:
            intersect = set.intersection(set(intersect), set(champion_lists[cluster_index][term]))
        else:
            intersect = []
            break

    distance = {}
    for filepath in intersect:
        doc = np.array(tfidf[cluster_index][filepath])
        query = np.array(qvector[cluster_index])
        if (np.dot(doc, query) != 0):
            distance[filepath] = spatial.distance.cosine(doc, query)

    x = []
    for i in distance.values():
        x.append(i)
    hq.heapify(x)

    key_list = list(distance.keys())
    val_list = list(distance.values())
    results = {}
    for i in range(k if k <= len(x) else len(x)):
        min = hq.heappop(x)
        results[key_list[val_list.index(min)]] = 1 - min

    answer = []
    for doc in results:
        record = [doc[doc.rfind("\\") + 1:len(doc)-4], results[doc]]
        answer.append(record)
    print(tabulate(answer, headers=['Document', 'Cosine Similarity'], tablefmt='pretty', stralign="center",
                   numalign="center"))

# Command-Line Interface
command = ""
while not command == "exit":
    print(">>", end="")
    command = input()
    if command == "exit":
        break
    elif command == "search":
        print("  Enter query: ", end="")
        search_query(5)
    elif command == "test":
        print("  Enter a single term: ", end="")
        test()
    elif command == "help":
        print("Enter \"search\" or \"test\".")
    else:
        print("Error: undefined command. Enter \"search\" or \"test\".")
