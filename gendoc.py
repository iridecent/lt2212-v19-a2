import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

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

def open_and_read(file):
    with open("textfile", "r", encoding="utf8") as f:
        text = f.read()
        wordlist = text.split(" ")
    return wordlist

def to_lower(wordlist):
    words = []
    for w in wordlist:
        s = w.lower()
        words.append(s)
    return words

def most_common(words, nr=10):
    return Counter(words).most_common(nr)

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


#python3 gendoc.py [-T|--tfidf] [-Sn|--svd n] [-Bm | --base-vocab m] foldername outputfile.txt


#preprocess, lowercase, remove punctuations
#create dictionary with words in all documents          sklearn countvectorizer
#fill in  dictionary for each document
#turning into list
#put list into panda
#transform panda np.array on dataframe of panda


#have list and dictionary at the same time: [3,5,6, ..., 7, ..] dict: index of list -> token and dict: tokens -> index