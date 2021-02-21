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


data_path = 'docs\\'
f = []
inverted_index = {}
champion_lists = {}

for filename in glob.iglob(data_path + '**/*.txt', recursive=True):
    f.append(os.path.abspath(filename))

term_frequency = {}
for filepath in f:
    file = open(filepath, 'r', encoding='utf-8')
    lines = file.readlines()
    for line in lines:
        exclude = set(string.punctuation).union(
            ['،', '؟', '!', '«', '»', '؛', '١', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۰']).union(string.digits)
        line = ''.join(ch for ch in line if ch not in exclude)
        terms = line.split()
        for term in terms:
            term = repetitive_stemming(term)
            if term in term_frequency:
                term_frequency[term] = term_frequency[term] + 1
            else:
                term_frequency[term] = 1

res = dict(sorted(term_frequency.items(), key=itemgetter(1), reverse=True)[:10])
stop = list(res.keys())

for filepath in f:
    file = open(filepath, 'r', encoding='utf-8')
    lines = file.readlines()
    pos = 0
    for line in lines:
        exclude = set(string.punctuation).union(
            ['،', '؟', '!', '«', '»', '؛', '١', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۰']).union(string.digits)
        line = ''.join(ch for ch in line if ch not in exclude)
        terms = line.split()
        for term in terms:
            term = repetitive_stemming(term)
            if term in stop:
                continue
            pos = pos + 1
            if term not in inverted_index:
                tmp = {}
                tmp[filepath] = []
                tmp[filepath].append(pos)
                inverted_index[term] = tmp
            else:
                if filepath in inverted_index[term]:
                    inverted_index[term][filepath].append(pos)
                else:
                    inverted_index[term][filepath] = []
                    inverted_index[term][filepath].append(pos)

od = collections.OrderedDict(sorted(inverted_index.items()))

for term in od:
    od[term] = collections.OrderedDict(sorted(od[term].items()))

tfidf = {}
tfs = {}
n = len(f)
def calculate_tfidf():
    for filepath in f:
        tmp = []
        for term in od:
            if filepath in od[term]:
                df = len(od[term])
                tf = len(od[term][filepath])
                w = math.log10(1 + tf) * math.log10(n / df)
                tmp.append(w)
            else:
                tmp.append(0)
        tfidf[filepath] = tmp

def calculate_tfs():
    for term in od:
        tmp = []
        for filepath in f:
            if filepath in od[term]:
                tmp.append(len(od[term][filepath]))
            else:
                tmp.append(0)
        tfs[term] = tmp

def calculate_champion_lists():
    for term in od:
        r = 2
        docs = {}
        i = 0
        for tf in tfs[term]:
            docs[f[i]] = tf
            i = i + 1
        top_list = []
        for j in range(r):
            max = -1
            max_doc = None
            for h in docs.keys():
                if docs[h] > max:
                    max = docs[h]
                    max_doc = h
            top_list.append(max_doc)
            del docs[max_doc]
        champion_lists[term] = top_list

calculate_tfidf()
calculate_tfs()
calculate_champion_lists()

def test():
    term = input()
    if (term in od):
        result = od[term]
        j = 0
        print(term)
        print("│")
        for doc in result:
            i = 0
            print("│_____" + doc[doc.rfind("\\") + 1:len(doc)])
            for position in od[term][doc]:
                if (j < len(result) - 1):
                    print("│     ", end="")
                else:
                    print("      ", end="")
                print("│_____" + str(position))
                i = i + 1
            if (j < len(result) - 1):
                print("│")
            j = j + 1
    else:
        print("Not Found!")


def search_query():
    query = input()
    terms = query.split()
    if len(terms) == 1:
        # stem query
        if query not in stop:
            query = repetitive_stemming(query)
            if query in od:
                result = od[query]
                answer = []
                for doc in result:
                    record = [doc[doc.rfind("\\") + 1:len(doc)]]
                    answer.append(record)
                print(tabulate(answer, headers=['Results'], tablefmt='pretty', stralign="center",
                               numalign="center"))
            else:
                print('No results found!')
            print()
        else:
            print('Query is empty!')
    else:
        result = {}
        c = 0
        for term in terms:
            # stem term
            if term not in stop:
                term = repetitive_stemming(term)
                if term in od:
                    for doc in od[term]:
                        if doc in result:
                            result[doc] = result[doc] + 1
                        else:
                            result[doc] = 1
            else:
                c = c + 1
        if c == len(terms):
            print('Query is empty!')
        else:
            ordered_result = dict(sorted(result.items(), key=lambda item: -item[1]))
            answer = []
            for doc in ordered_result:
                ordered_result[doc] = ordered_result[doc] / (len(terms) - c)
                record = [doc[doc.rfind("\\") + 1:len(doc)], ordered_result[doc]]
                answer.append(record)
            print(tabulate(answer, headers=['Document', 'Score'], tablefmt='pretty', stralign="center",
                           numalign="center"))

def search_query2(k):
    query = input()
    tmp = query.split()
    terms = []
    for term in tmp:
        stemmed = repetitive_stemming(term)
        if stemmed not in stop:
            terms.append(stemmed)

    qvector = []
    for term in od:
        if term in terms:
            df = len(od[term])
            tf = terms.count(term)
            w = math.log10(1 + tf) * math.log10(n / df)
            qvector.append(w)
        else:
            qvector.append(0)

    intersect = []
    for filepath in f:
        intersect.append(filepath)

    # for term in terms:
    #     intersect = set.intersection(set(intersect), set(champion_lists[term]))

    distance = {}
    for filepath in intersect:
        doc = np.array(tfidf[filepath])
        query = np.array(qvector)
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
        record = [doc[doc.rfind("\\") + 1:len(doc)], results[doc]]
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
        search_query()
    elif command == "search2":
        print("  Enter query: ", end="")
        search_query2(5)
    elif command == "test":
        print("  Enter a single term: ", end="")
        test()
    elif command == "help":
        print("Enter \"search\" or \"test\".")
    else:
        print("Error: undefined command. Enter \"search\" or \"test\".")