# By Ihsaan Malek and Olivier Racette
# The goal of this assignment is to build a Naive Bayes Bag-Of-Word model to determine if a tweet contains factual claims about covid-19
# Data files are tab seperated

import argparse
import csv
import string
from pathlib import Path

#external dependencies
#import numpy as np
#from sklearn.metrics import ...            #we probably can't import this for A3

data_folder = Path("../data/")

#Reads the passed delimieter seperated file to a list of dictionaries with keys as col_names. 
def read_data(file, delim='\t', col_names=None):
    result = []

    with open(file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delim, fieldnames=col_names)

        for row in reader:
            result.append(row)

    return result

#numpy really doesnt like the tsv files we have :(
#it's not using the delimiter properly even though it is clearly defined
#might be due to bad characters in the files? checking them out in notepad++ with utf-8 shows some bizarre things
#I guess I'll stick to default csv library
#def read_training_np(file):
#    data = np.genfromtxt(file, dtype=None, delimiter="\t", encoding='utf8', names=True)

#    return data



#Sets and retrieves the command line arguments
#Current arguments:
    #training data file, default "covid_training.tsv"
    #test data file, default "covid_test_public.tsv"
#Returns a Namespace object
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--train", type=str, help="Name of the training data file.", default="covid_training.tsv")
    parser.add_argument("-s", "--test", type=str, help="Name of the test data file.", default="covid_test_public.tsv")

    return parser.parse_args()


# Builds a vocabulary out of the data dictionary, along with a frequency count for each word. 
# Assumes data is a list of dictionaries that containt "text" entry.
# Vocabulary can be filtered: entries with a frequency less than 2 are removed.
#NOTE: need to "clean" the words of any punctuation
#NOTE: no smoothing yet
def build_vocabulary(data, filter_vocab=False, smooth=0.01):
    vocab = {}
    
    for row in data:
        for word in row["text"].lower().split():
            if word not in vocab:
                vocab.update({word:1})
            else:
                vocab[word] += 1

    if filter_vocab:
        filtered_vocab = {}

        for word in vocab:
            if vocab[word] >= 2:
                filtered_vocab.update({word:vocab[word]})

        vocab = filtered_vocab

    return vocab



def run():
    args = get_args()

    train_set = read_data(data_folder / args.train)
    test_set = read_data(data_folder / args.test, col_names=("tweet_id", "text"))

    train_vocab = build_vocabulary(train_set)
    test_vocab = build_vocabulary(test_set)

    train_vocab_filtered = build_vocabulary(train_set, True)
    test_vocab_filtered = build_vocabulary(test_set, True)

    #use the words as features and word frequencies as feature values

    #for k,v in train_vocab.items():
    #    print(k, "\t", str(v))

    #print("============================")

    #for k,v in test_vocab.items():
    #    print(k, "\t", str(v))


if __name__ == "__main__":
    run()