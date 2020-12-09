# By Ihsaan Malek and Olivier Racette
# The goal of this assignment is to build a Naive Bayes Bag-Of-Word model to determine if a tweet contains factual claims about covid-19
# Data files are tab seperated

import argparse
import csv
import string
from pathlib import Path
from math import log10

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


# Builds a vocabulary out of the data dictionary for label "label_name" with value "label_value", along with a frequency count for each word. 
# Assumes data is a list of dictionaries that containt "text" entry.
# Vocabulary can be filtered: entries with a frequency less than 2 are removed.
# No need to clean the words
def build_vocabulary(data, label_name="q1_label", label_value="yes", filter_vocab=False):
    vocab = {}
    
    for row in data:
        if row[label_name] == label_value:
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


#Returns the conditional in log10, p(x) = (count(x) + smooth) /  (count(all words) + size_of_vocab*smooth)
#Assumes vocab contains the frequency of each word
def calc_conditionals(vocab, smooth=0.01):
    condi = dict.fromkeys(vocab.keys())

    word_count = sum(vocab.values())

    for word in condi:
        condi[word] = log10((vocab[word] + smooth) / (word_count + len(vocab)*smooth))

    return condi


#Returns the priors for the given dataset and label in log10
#Label defaults to "q1_label" and label_value to "yes" as per the assignment guidelines
#Does not handle div by 0 or log(0)
def calc_priors(dataset, label_name="q1_label", label_value="yes"):
    denominator = len(dataset)

    numerator = 0

    for row in dataset:
        if row[label_name] == label_value:
            numerator += 1

    return log10(numerator / denominator)


def run():
    args = get_args()

    train_set = read_data(data_folder / args.train)
    test_set = read_data(data_folder / args.test, col_names=("tweet_id", "text", "q1_label"))

    #Training
    train_vocab = build_vocabulary(train_set)
    train_vocab_filtered = build_vocabulary(train_set, filter_vocab=True)

    train_conditionals = calc_conditionals(train_vocab)
    train_conditionals_filtered = calc_conditionals(train_vocab_filtered)

    train_priors = calc_priors(train_set)

    #Testing
    test_vocab = build_vocabulary(test_set)
    test_vocab_filtered = build_vocabulary(test_set, filter_vocab=True)


if __name__ == "__main__":
    run()