import re
import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from collections import defaultdict


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
parser.add_argument("vectorfile1", type=str, default=None,
                     help="The name of the csv file of the first topic to calculate cosine similarity.")

parser.add_argument("vectorfile2", type=str, default=None,
                     help="The name of the csv file of the second topic to calculate cosine similarity.")

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
            vectors[topic+" "+file] = counts              # concatenate subfoldername and filename
    return vectors




def get_data_for_matrix(vectors):
    """Modify dictionary of vectors to fit pandas DataFrame"""
    all_data = vectors
    all_columnlabels = []
    for k, v in sorted(vectors[list(vectors.keys())[0]].items()):   # get tokennames out of random file - tokens are the same for each file
        all_columnlabels.append(k)
    all_rowlabels = []
    for filename in sorted(vectors.keys()):
        all_rowlabels.append(filename)                          # get filenames as rowlabels
        rows = []
        for word, count in sorted(vectors[filename].items()):       # remove wordnames and create list only containing counts in the same order as the columnlabels
            rows.append(count)
        all_data[filename] = rows                               # map list of integers (vector) to corresponding file
    return all_data, all_columnlabels, all_rowlabels


def remove_duplicate_vectors(all_data, all_columnlabels, all_rowlabels):
    """Identifies and removes duplicate vectors"""
    data = {}
    for k, v in all_data.items():
        data[k] = tuple(v)                                      # turn vector into tuple to make it non-hashable
    upside_down = defaultdict(list)
    for name, vect in data.items():
        upside_down[vect].append(name)                          # vector as key and append all article names to the value with same vector
    dropped_articles = []
    for vec in upside_down.keys():
        if len(upside_down[vec])>1:                             # check if there are more than one article with particular vector
            for art in upside_down[vec][1:]:
                del data[art]                                   # delete all article vectors after the first one with the same
                dropped_articles.append(art)                    # keep track of dropped articles
                all_rowlabels.remove(art)
    columnlabels = all_columnlabels                             # columns stay the same
    rowlabels = sorted(data.keys())
    return data, columnlabels, rowlabels, dropped_articles



def create_term_document_matrix(data, columns):
    """Feed dictionary into DataFrame"""
    matrix = pd.DataFrame.from_dict(data, orient="index", columns=columns)
    return matrix


def apply_tfidf(matrix):
    """Transform raw counts to tfidf values"""
    tfidf = TfidfTransformer().fit_transform(matrix)
    tfidf_values = tfidf.toarray()
    return tfidf_values

def create_tfidf_matrix(tfidf_values, columnlabels,rowlabels):
    """Feed tfidf values into DataFrame"""
    tfidf_matrix = pd.DataFrame(tfidf_values)
    tfidf_matrix.columns = [columnlabels]
    tfidf_matrix.index = [rowlabels]
    tfidf_dict = {}                                         #transform array into dictionary to be used in separate topic function
    tfidf_values = np.array(tfidf_values).tolist()
    for i, label in enumerate(rowlabels):
        tfidf_dict[label] = tfidf_values[i]
    return tfidf_matrix, tfidf_dict


def get_raw_numpy_array(matrix):
    """Get raw np array which is needed for the TruncatedSVD function"""
    raw_array = matrix.values
    return raw_array


def create_SVD(array, n, rowlabels):
    """Feed raw array into TruncatedSVD function. Outputs SVD matrix of dimensionality n"""
    svd = TruncatedSVD(n_components=n)
    svdm = svd.fit_transform(array)
    svd_matrix = pd.DataFrame(svdm)
    svd_matrix.index = [rowlabels]
    svd_dict = {}
    svdm = np.array(svdm).tolist()
    for i, label in enumerate(rowlabels):
        svd_dict[label] = svdm[i]
    return svd_matrix, svd_dict


def separate_topics(folder, data_dict):
    """Separate the two different topics to be able to create two different csv files"""
    topic1, topic2 = os.listdir(folder)
    vect1 = []
    vect2 = []
    for k,v in data_dict.items():
        if topic1 in k:
            vect1.append(v)
        else:
            vect2.append(v)
    vectors1 = pd.DataFrame(vect1)
    vectors2 = pd.DataFrame(vect2)
    return vectors1, vectors2



print("Loading data from directory {}.".format(args.foldername))


if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.tfidf:
    print("Applying tf-idf to raw counts.")

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))


print("Writing matrix to {} and producing {} and {}.".format(args.outputfile, args.vectorfile1, args.vectorfile2))


vectors = create_vectors(args.foldername, args.basedims)
all_data, all_columnlabels, all_rowlabels = get_data_for_matrix(vectors)
data, columnlabels, rowlabels, dropped_articles = remove_duplicate_vectors(all_data, all_columnlabels, all_rowlabels)
matrix = create_term_document_matrix(data, columnlabels)
tfidf_values = apply_tfidf(matrix)
raw_array = get_raw_numpy_array(matrix)





print("The following articles got dropped: ", dropped_articles)

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)
pd.set_option("display.width", 10**1000)

if args.vectorfile1 and not args.vectorfile2:
    print("Please enter two csv output files, one for each topic!")

if not args.vectorfile1 and args.vectorfile2:
    print("Please enter two csv output files, one for each topic!")

if args.basedims and args.svddims:
    if args.basedims < args.svddims:
        print("The number of SVD dimensions must be bigger than the number of used word types.")

if not args.tfidf and not args.svddims:
    vectors1, vectors2 = separate_topics(args.foldername, data)
    vectors1.to_csv(args.vectorfile1, header=None, index=None)
    vectors2.to_csv(args.vectorfile2, header=None, index=None)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(matrix)

if args.tfidf and not args.svddims:
    tfidf_matrix, tfidf_dict = create_tfidf_matrix(tfidf_values, columnlabels, rowlabels)
    vectors1, vectors2 = separate_topics(args.foldername, tfidf_dict)
    vectors1.to_csv(args.vectorfile1, header=None, index=None)
    vectors2.to_csv(args.vectorfile2, header=None, index=None)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(tfidf_matrix)

if not args.tfidf and args.svddims:
    svd_matrix, svd_dict = create_SVD(raw_array, args.svddims, rowlabels)
    vectors1, vectors2 = separate_topics(args.foldername, svd_dict)
    vectors1.to_csv(args.vectorfile1, header=None, index=None)
    vectors2.to_csv(args.vectorfile2, header=None, index=None)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(svd_matrix)

if args.tfidf and args.svddims:
    tfidfsvd_matrix, tfidfsvd_dict = create_SVD(tfidf_values, args.svddims, rowlabels)
    vectors1, vectors2 = separate_topics(args.foldername, tfidfsvd_dict)
    vectors1.to_csv(args.vectorfile1, header=None, index=None)
    vectors2.to_csv(args.vectorfile2, header=None, index=None)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        sys.stdout = open(args.outputfile, "w")
        print(tfidfsvd_matrix)
