import re
import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import pprint
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
                nopunct = re.sub(r"\d+|[!,\.\*\-\–"
                                 r":;%&?€\+#@£$∞§"
                                 r"\|\/\[\]\(\)\{\}\""
                                 r"\„\“\'\´«»\n]+","",
                                 text.lower())          # preprocessing: strip punctuation and special characters
                words = nopunct.split(" ")              # preprocessing: tokenization
                allwords += words
    vocabulary = Counter(allwords).most_common(m)       # if -B is set takes m most common words
    vocabulary.pop(0)                                   # remove empty string
    vectorspace = {}
    for entry in vocabulary:                            # vocabulary is a list of two tuples each containing word and count
        vectorspace[entry[0]] = 0                       # set all values to 0 to get a blueprint for the countvectors of the files
    return vectorspace


def create_vectors(folder, m=None):
    """Creating feature (i.e. count) vectors for every document"""
    vectors = {}
    for topic in os.listdir(folder):
        subfolder_path = os.path.join(folder, topic)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            counts = create_vectorspace(folder, m)         # initialize vectorspace dict with zero-counts for every file
            with open(file_path, "r", encoding="utf8") as f:
                text = f.read()
                nopunct = re.sub(r"\d+|[!,\.\*\-\–"
                                 r":;%&?€\+#@£$"
                                 r"∞§\|\/\[\]\(\)\{"
                                 r"\}\"\„\“\'\´«»\n]+","", text.lower())
                words = nopunct.split(" ")
                for word in words:
                    if word not in counts:                 # if word was filtered by -B m is is deleted
                        del word
                    else:
                        counts[word] += 1                  # vectorspace dict is updated by count of word in that file
            vectors[topic+" "+file] = counts               # concatenate subfoldername and filename
    return vectors



def get_data_for_matrix(vectors):
    """Modify dictionary of vectors to fit pandas DataFrame"""
    data = vectors
    columnlabels = []
    for k, v in vectors[list(vectors.keys())[0]].items():   # get tokennames out of random file - tokens are the same for each file
        columnlabels.append(k)
    rowlabels = []
    for filename in vectors.keys():
        rowlabels.append(filename)                          # get filenames as rowlabels
        rows = []
        for word, count in vectors[filename].items():       # remove wordnames and create list only containing counts in the same order as the columnlabels
            rows.append(count)
        data[filename] = rows                               # map list of integers (vector) to corresponding file
    return data, columnlabels, rowlabels                    #todo: remove dublicate articles


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


def get_raw_numpy_array(matrix):                            # TrucatedSVD takes a raw array
    raw_array = matrix.values
    return raw_array


def create_SVD(array, n):
    svd = TruncatedSVD(n_components=n)
    svdmatrix = svd.fit_transform(array)
    return svdmatrix


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


vectors = create_vectors(args.foldername, args.basedims)
data, columnlabels, rowlabels = get_data_for_matrix(vectors)
matrix = create_term_document_matrix(data, columnlabels)
tfidf = apply_tfidf(matrix)
tfidf_matrix = create_tfidf_matrix(tfidf, columnlabels, rowlabels)
raw_array = get_raw_numpy_array(matrix)

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)
pd.set_option("display.width", 10**100000)


if not args.tfidf and not args.svddims:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(matrix)

if args.tfidf and not args.svddims:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(tfidf_matrix)

if not args.tfidf and args.svddims:
    svd = create_SVD(raw_array, args.svddims)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(svd)

if args.tfidf and args.svddims:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(create_SVD(tfidf, args.svddims))
