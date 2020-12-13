# By Ihsaan Malek and Olivier Racette
# The goal of this assignment is to build a Naive Bayes Bag-Of-Word model to determine if a tweet contains factual claims about covid-19
# Data files are tab seperated

import argparse
import csv
import string
from pathlib import Path
from math import log10

#user imports
from model_eval import evaluate, output_trace
from sanitizer import sanitize

data_folder = Path("../data/")
output_folder = Path("../output/")


#Reads the passed delimieter seperated file to a list of dictionaries with keys as col_names. 
def read_data(file, delim='\t', col_names=None):
    result = []

    with open(file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delim, fieldnames=col_names)

        for row in reader:
            result.append(row)

    return result


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
def build_vocabulary(data, label_name="q1_label", filter_vocab=False):
    vocab = {}

    for row in data:
        for word in row["text"].lower().split():
            if row[label_name] == "yes":
                label_value = "yes"
            else:
                label_value = "no"

            if word not in vocab:
                vocab.update({
                    word: {"yes": 0, "no": 0}
                })
            
            vocab[word][label_value] += 1

    if filter_vocab:
        filtered_vocab = {}

        for word in vocab:
            if vocab[word]["yes"] + vocab[word]["no"] >= 2:
                filtered_vocab[word] = vocab[word]

        vocab = filtered_vocab

    return vocab


#Returns the conditional in log10, p(x) = (count(x) + smooth) /  (count(all words) + size_of_vocab*smooth)
#Assumes vocab contains the frequency of each word
def calc_conditionals(vocab, label_value, smooth=0.01):
    condi = dict.fromkeys(vocab.keys())

    word_count = 0  

    for word in vocab:
        word_count += vocab[word][label_value]

    for word in condi:
        condi[word] = log10((vocab[word][label_value] + smooth) / (word_count + len(vocab)*smooth))

    return condi


#Returns the priors for the given dataset and label in log10
#Label defaults to "q1_label" and label_value to "yes" as per the assignment guidelines
#Does not handle div by 0 or log(0)
def calc_priors(dataset, label_value, label_name="q1_label"):
    denominator = len(dataset)

    numerator = 0

    for row in dataset:
        if row[label_name] == label_value:
            numerator += 1

    return log10(numerator / denominator)


def correct_helper(value1, value2):
    if value1 == value2:
        bool_Str ='correct'
    else:
        bool_Str ='wrong'
        
    return bool_Str


#Our Naive Bayes implementation
#Builds vocabulary, conditionals and priors for each label with the training set
#Then uses these values to calculate the scores for each test set tweet.
#NOTE: Words that are not found in the vocabulary are ignored
def Naive_Bayes(train_dataset, test_dataset, label_values, filtered, smooth = 0.01):
    #Binomial Naive Bayes
    result = {}
    
    train_vocab =  build_vocabulary(train_dataset, filter_vocab=filtered)

    train_conditionals1 = calc_conditionals(train_vocab, label_values[0])
    train_priors1 = calc_priors(train_dataset, label_value=label_values[0])
    
    train_conditionals2 = calc_conditionals(train_vocab, label_values[1])
    train_priors2 = calc_priors(train_dataset, label_value=label_values[1])

    
    for row in test_dataset: 
        #tweet score = prior + condi1 + condi2 + ... + condiN
        prob1 = train_priors1 
        prob2 = train_priors2
        
        for word in row["text"].lower().split():
            try:
                prob1 += train_conditionals1[word]
            except:
                pass

            try:       
                prob2 += train_conditionals2[word]
            except:
                pass

        #formatting the return dictionary    
        if prob1 > prob2:              
            prob_Str= '{:.1e}'.format(prob1)
            eval_label = correct_helper(label_values[0],row["q1_label"])
            result[row['tweet_id']] = [label_values[0], prob_Str, row["q1_label"],eval_label]  
        else:     
            prob_Str= '{:.1e}'.format(prob2)
            eval_label = correct_helper(label_values[1],row["q1_label"])
            result[row['tweet_id']] = [label_values[1], prob_Str, row["q1_label"],eval_label]

    return result


#Reads data from training set and test set
#Runs naive bayes on both, unfiltered and filtered
#Outputs trace and evaluation files
def run():
    args = get_args()

    train_set = read_data(data_folder / args.train)
    test_set = read_data(data_folder / args.test, col_names=("tweet_id", "text", "q1_label"))

    regular_solution = Naive_Bayes(train_set,test_set, ['yes','no'], False)
    filtered_solution = Naive_Bayes(train_set,test_set, ['yes','no'], True)

    output_trace(output_folder / "trace_NB-BOW-OV.txt", regular_solution)
    output_trace(output_folder / "trace_NB-BOW-FV.txt", filtered_solution)
    
    evaluate(output_folder / "eval_NB-BOW-OV.txt", regular_solution, "yes", "no")
    evaluate(output_folder / "eval_NB-BOW-FV.txt", filtered_solution, "yes", "no")


    #using sanitized input
    train_set_cleaned = sanitize(train_set)
    test_set_cleaned = sanitize(test_set)

    regular_solution = Naive_Bayes(train_set_cleaned,test_set_cleaned, ['yes','no'], False)
    filtered_solution = Naive_Bayes(train_set_cleaned,test_set_cleaned, ['yes','no'], True)

    output_trace(output_folder / "trace_NB-BOW-OV_clean.txt", regular_solution)
    output_trace(output_folder / "trace_NB-BOW-FV_clean.txt", filtered_solution)
    
    evaluate(output_folder / "eval_NB-BOW-OV_clean.txt", regular_solution, "yes", "no")
    evaluate(output_folder / "eval_NB-BOW-FV_clean.txt", filtered_solution, "yes", "no")


if __name__ == "__main__":
    run()
