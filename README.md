# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Iris Epple

## Additional instructions

gendoc: 
2 additional optional arguments: Two csv files, one for each topic, to use as input for simdoc. Name the files
according to the subfolders and in alphabetical order.

simdoc: takes two csv files as input

Reduced the number of articles to 300 in each topic.


## Results and discussion



### Vocabulary restriction.

I chose 20, to get a few more than the obvious ones (i.e. the stop words).

### Result table

| Files     | cs crude | cs grain | cross cs c-g | cross cs g-c |
|-----------|----------|----------|--------------|--------------|
| output 1  |  0.36    | 0.33     | 0.30         | 0.30         |
| output 2  |  0.67    | 0.61     | 0.59         | 0.59         |
| output 3  |  0       | 0        | 0            | 0            |
| output 4  |  0.00018 | 0        | 0            | 0            |
| output 5  |  0.53    | 0.54     | 0.48         | 0.48         |
| output 6  |  0.48    | 0.49     | 0.43         | 0.43         |
| output 7  |  0.36    | 0.33     | 0.30         | 0.30         |
| output 8  |  0.36    | 0.33     | 0.30         | 0.30         |

### The hypothesis in your own words

I am not sure about my numbers, since  cosine similarity of zero between two documents 
means that they have no words in common. (see output 3 & 4) 
Duplicate documents where removed, therefore there be
any identical vectors. But most of the vectors, at least within the 
same topic should have some words in common (which is e.g. the stopwords). 
But as tf-ifd is applied, the impact of words with very high and very low frequency is reduced.
This means that after applying tf-idf to the raw counts, the value for each word is different 
in every document, even for two documents which had before for example a count of 15 for "a", 
since the whole document is taken into account. 
Against this background it is not too surprising anymore that there are apparently no two
vectors with the same value in one particular position.
 
SVD truncation seems to make no difference  (compared to raw counts - output 1 vs 7 & 8, 
which could be due to the nature of SVD - i.e redistributing features and values to yield the 
demanded dimension, but with keeping the relative distributions and differences

### Discussion of trends in results in light of the hypothesis

see above