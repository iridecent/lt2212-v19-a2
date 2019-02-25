import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile1", type=str,
                    help="The name of the input  file for the matrix data.")

parser.add_argument("vectorfile2", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()


data1 = np.loadtxt(args.vectorfile1, dtype='i', delimiter=',')
data2 = np.loadtxt(args.vectorfile2, dtype='i', delimiter=',')


print(data1, data2)



def compute_cosine_similarities(array1, array2=None):
    similarities = []
    for index, v1 in enumerate(array1):
        if array2 is not None:
            for v2 in array2:
                similarities.append(cosine_similarity([v1], [v2]))
        else:
            for v2 in array1[index + 1:]:
                similarities.append(cosine_similarity([v1], [v2]))
    average_similarity = sum(similarities)/len(similarities)
    return average_similarity


topic1_same = compute_cosine_similarities(data1)
topic2_same = compute_cosine_similarities(data2)

topic1_to_topic2 = compute_cosine_similarities(data1, data2)
topic2_to_topic1 =compute_cosine_similarities(data2, data1)


print("Reading matrix from {}.".format(args.vectorfile1))
print("Reading matrix from {}.".format(args.vectorfile2))


print("Average cosine similarity within same topic, topic 1: ", topic1_same)

print("Average cosine similarity within same topic, topic 2: ", topic2_same)

print("Average cosine similarity to other topic (topic 1 to topic 2): ", compute_cosine_similarities(data1, data2))
print("Average cosine similarity to other topic (topic 2 to topic 1): ", compute_cosine_similarities(data2, data1))


