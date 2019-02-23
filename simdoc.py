import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()


def get_raw_numpy_array(matrix):
    raw_array = matrix.values
    return raw_array


data = np.loadtxt(args.vectorfile, dtype='i', delimiter=',')
print(data)



print("Reading matrix from {}.".format(args.vectorfile))



