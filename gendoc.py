import re
import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter

# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()



def create_vectorspace(folder, m=None):
    """Creating a dictionary containing all occuring words in all documents as keys"""
    allwords = []
    for topic in os.listdir(folder):
        subfolder_path = os.path.join(folder, topic)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            with open(file_path, "r", encoding="utf8") as f:
                text = f.read()
                nopunct = re.sub(r"\d+|[!,\.\*\-\–:;%&?€\+#@£$∞§\|\/\[\]\(\)\{\}\"\„\“\'\´«»\n]+","", text.lower())
                words = nopunct.split(" ")
                allwords += words
    vocabulary = Counter(allwords).most_common(m)
    vocabulary.pop(0)                                        # remove empty word
    vectorspace = {}
    for entry in vocabulary:
        vectorspace[entry[0]] = 0
    return vectorspace


def create_vectors(folder, m=None):
    """Creating feature vectors for every document"""
    vectors = {}
    for topic in os.listdir(folder):
        subfolder_path = os.path.join(folder, topic)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            counts = create_vectorspace(folder, m)
            with open(file_path, "r", encoding="utf8") as f:
                text = f.read()
                nopunct = re.sub(r"\d+|[!,\.\*\-\–:;%&?€\+#@£$∞§\|\/\[\]\(\)\{\}\"\„\“\'\´«»\n]+","", text.lower())
                words = nopunct.split(" ")
                for word in words:
                    if word not in counts:
                        del word
                    else:
                        counts[word] += 1
            vectors[topic+" "+file] = counts
    return vectors

def get_data_for_matrix(vectors):
    """Modify dictionary of vectors to fit pandas DataFrame"""
    data = vectors
    columnlabel = []
    for k, v in vectors[list(vectors.keys())[0]].items():
        columnlabel.append(k)
    rowlabel = []
    for filename in vectors.keys():
        rowlabel.append(filename)
        rows = []
        for word, count in vectors[filename].items():
            rows.append(count)
        data[filename] = rows
    return data, columnlabel, rowlabel


def create_term_document_matrix(data, columns):
    matrix = pd.DataFrame.from_dict(data, orient="index", columns=columns)
    return matrix


def apply_tfidf(matrix):
    tfidf = TfidfTransformer().fit_transform(matrix)
    tfidf = tfidf.toarray()
    return tfidf

def create_tfidf_matrix(tfidf, columnlabels,rowlabels):
    tfidf_matrix = pd.DataFrame(tfidf)
    tfidf_matrix.columns = [columnlabels]
    tfidf_matrix.index = [rowlabels]
    return tfidf_matrix


def get_raw_numpy_array(matrix):        #for svd
    raw_array = matrix.values
    return raw_array


def create_SVD(array, n):
    svd = TruncatedSVD(n_components=n)
    svdmatrix = svd.fit_transform(array)
    return svdmatrix





vectors = create_vectors(args.foldername, args.basedims)
data, columnlabels, rowlabels = get_data_for_matrix(vectors)
matrix = create_term_document_matrix(data, columnlabels)

#print(data, columnlabels, rowlabels)
#print(matrix)

array = get_raw_numpy_array(matrix)
#tfidf = apply_tfidf(matrix)
#print(tfidf)
#print(create_tfidf_matrix(tfidf, columnlabels, rowlabels))


print(create_SVD(array,args.svddims))


print("Loading data from directory {}.".format(args.foldername))



if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.tfidf:
    print("Applying tf-idf to raw counts.")

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.

print("Writing matrix to {}.".format(args.outputfile))





#preprocess, lowercase, remove punctuations
#create dictionary with words in all documents
# fill in  dictionary for each document

#turning into list

#put list into panda
#transform panda np.array on dataframe of panda


#have list and dictionary at the same time: [3,5,6, ..., 7, ..] dict: index of list -> token and dict: tokens -> index