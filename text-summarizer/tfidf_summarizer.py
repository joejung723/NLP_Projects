#!/usr/bin/env python3
# # coding: utf-8
""" Text Summarizer using tf x idf """

__author__="Joe Jung"


import sys
import re
import math
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


if len(sys.argv) > 1:
    input_file = sys.argv[1]
    output_file = 'final_summary.txt'
else:
    input_file = 'testfile.txt'
    # input_file = 'thecow.txt'
    # input_file = 'football.txt'
    # input_file = 'columbus.txt'
    output_file = 'final_summary.txt'

stop_words = set(stopwords.words('english'))
print(stop_words)

original_sentences = ''

dct_cos_sim = {}

# sklearn vectorizer
vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectorizer = TfidfVectorizer(stop_words = stop_words)


def main():

    with open(input_file, 'r', encoding='utf8') as reader:
        s = reader.read()
        tokenized_sent = nltk.sent_tokenize(s)
        print(tokenized_sent)
        V = vectorizer.fit_transform(tokenized_sent)
        print(vectorizer.get_feature_names())
        print(V.toarray())
        TFIDF = tfidf_vectorizer.fit_transform(tokenized_sent)
        print(TFIDF)
        print('TFIDF to array: ', TFIDF.toarray())
        pd_dataf = pd.DataFrame(TFIDF.toarray(), columns=tfidf_vectorizer.get_feature_names())
        print(tfidf_vectorizer.get_feature_names())
        print(pd_dataf)

        v_matrix = V.toarray()
        tfidf_matrix = TFIDF.toarray()
        print(tfidf_matrix)

        # for i in range(len(v_matrix)):
        #
        #     l1 = v_matrix[i]
        #
        #     for j in range(len(v_matrix)):
        #
        #         if j < i or j == i:
        #             continue
        #
        #         key = i, j
        #
        #         if key not in dct_cos_sim:
        #             dct_cos_sim[key] = 0
        #
        #         l2 = v_matrix[j]
        #         cos_sim = cosine_similarity(l1, l2)
        #         print(cos_sim)
        #         dct_cos_sim[key] = cos_sim

        # finds cosine similarity between sentences and puts the comparing sentences as key and its cosine similarity as value
        for i in range(len(tfidf_matrix)):

            l1 = tfidf_matrix[i]

            for j in range(len(tfidf_matrix)):

                if j < i or j == i:
                    continue

                key = i, j

                if key not in dct_cos_sim:
                    dct_cos_sim[key] = 0

                l2 = tfidf_matrix[j]
                cos_sim = cosine_similarity(l1, l2)
                print(cos_sim)
                dct_cos_sim[key] = cos_sim


        print(dct_cos_sim)

        # list of sentence indices
        final_summary = []

        # length of summary depending on percentage value
        size_of_summary = int(len(tokenized_sent) * .2)
        print(size_of_summary)

        top_sent_dict = dict(Counter(dct_cos_sim).most_common(size_of_summary))
        print(top_sent_dict)

        for k,v in top_sent_dict.items():
            final_summary.extend(k)

        # sort the sentences indices in order
        sorted_set = set(final_summary)
        print(sorted(sorted_set))

        output_summary = ''

        # print(type(tokenized_sent), tokenized_sent)
        for i in sorted_set:
            output_summary += '\n' + tokenized_sent[i]
            print(tokenized_sent[i])

        # writes final summary to final summary txt file
        with open(output_file, 'w', encoding='utf8') as file:
            file.write(output_summary)


def cosine_similarity(list1,list2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0

    for i in range(len(list1)):
        x = list1[i]; y = list2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    return sumxy / (math.sqrt(sumxx) + math.sqrt(sumyy))


if __name__ == "__main__":
    main()

    # test