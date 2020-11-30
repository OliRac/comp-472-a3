# By Ihsaan Malek and Olivier Racette
# The goal of this assignment is to build a Naive Bayes Bag-Of-Word model to determine if a tweet contains factual claims about covid-19
# Data files are tab seperated

import argparse
import csv
from pathlib import Path

#external dependencies
#import numpy as np
#from sklearn.metrics import ...            #we probably can't import this for A3

data_folder = Path("../data/")

#Reads the passed delimieter seperated file to a list of rows. 
#If specified, will output a list of dictionaries (useful for training set)
def read_data(file, to_dict=False, delim='\t'):
    result = []

    with open(file, encoding='utf-8') as f:
        if to_dict:
            reader = csv.DictReader(f, delimiter=delim)
        else:
            reader = csv.reader(f, delimiter=delim)

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



def run():
    args = get_args()

    train_set = read_data(data_folder / args.train, to_dict=True)
    test_set = read_data(data_folder / args.test)

if __name__ == "__main__":
    run()