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
output_folder = Path("../output/")

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
def build_vocabulary(data, label_value, label_name="q1_label", filter_vocab=False):
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
def calc_priors(dataset, label_value, label_name="q1_label"):
    denominator = len(dataset)

    numerator = 0

    for row in dataset:
        if row[label_name] == label_value:
            numerator += 1

    return log10(numerator / denominator)


#Returns scores for each entry in the dataset, for label_value and filtered or not
def calc_score(dataset, label_value, filtered, smooth = 0.01):
    vocab =  build_vocabulary(dataset, label_value=label_value, filter_vocab=filtered)
    conditionals = calc_conditionals(vocab, smooth=smooth)
    priors = calc_priors(dataset, label_value=label_value)

    #returning empty dict for now, replace with score calculation later
    return {}

def correct_helper(value1, value2):
    if value1 == value2:
        bool_Str ='correct'
    else:
        bool_Str ='wrong'
        
    return bool_Str

def Naive_Bayes(train_dataset, test_dataset, label_values, filtered, smooth = 0.01):
    #Binomial Naive Bayes
    result = {}
    
    train_vocab1 =  build_vocabulary(train_dataset, label_value=label_values[0], filter_vocab=filtered)
    train_conditionals1 = calc_conditionals(train_vocab1)
    train_priors1 = calc_priors(train_dataset, label_value=label_values[0])
    
    train_vocab2 =  build_vocabulary(test_dataset, label_value=label_values[1], filter_vocab=filtered)
    train_conditionals2 = calc_conditionals(train_vocab2)
    train_priors2 = calc_priors(test_dataset, label_value=label_values[1])
    
    noword_in1 = 0
    noword_in2 = 0
    
    for row in test_dataset:
        
        prob1 = 1 #prob needs to start at 1 because of the numerical identities that 1 has over 0
        prob2 = 1
        any_1 = 0 #flags to double check if inital probablity of 1 is changed
        any_2 = 0
        
        for word in row["text"].lower().split():
            try:
                word_prob1 = train_vocab1[word]/sum(train_vocab1.values()) #prob of word appearing in vocab
                prob1 = prob1*(train_conditionals1[word]/word_prob1)
                any_1 = 1
            except:
                noword_in1 += 1
                pass
            try:       
                word_prob2 = train_vocab2[word]/sum(train_vocab2.values())
                prob2 = prob2*(train_conditionals2[word]/word_prob2)
                any_2 = 1
            except:
                noword_in2 += 1
                pass
                     
            
        if any_1 == 1 and any_2 == 1: #probability sucessfully calculated for both classes
        
            prob1 = prob1*train_priors1
            prob2 = prob1*train_priors2
            
            if prob1 > prob2:
                
                prob_Str= '{:.1e}'.format(prob1)
                #print(prob1)
                eval_label = correct_helper(label_values[0],row["q1_label"])
                result[row['tweet_id']] = [label_values[0], prob_Str, row["q1_label"],eval_label]
            
            else:
                
                prob_Str= '{:.1e}'.format(prob2)
                #print(prob2)
                eval_label = correct_helper(label_values[1],row["q1_label"])
                result[row['tweet_id']] = [label_values[1], prob_Str, row["q1_label"]]
                
        elif any_1 == 1 and any_2 == 0:

            prob1 = prob1*train_priors1
            #print(prob1)
            prob_Str= '{:.1e}'.format(prob1)
            eval_label = correct_helper(label_values[0],row["q1_label"])
            result[row['tweet_id']] = [label_values[0], prob_Str, row["q1_label"]]
            
        else:
            
            prob2 = prob1*train_priors2
            #print(prob2)
            prob_Str= '{:.1e}'.format(prob2)
            eval_label = correct_helper(label_values[1],row["q1_label"])
            result[row['tweet_id']] = [label_values[1], prob_Str, row["q1_label"]]
    
    #print(noword_in1)
    #print(noword_in2)
    return result
            

#Outputs the result data dictionary to a file in the output directory. 
#Uses the following format: tweet_id[space][space]predicted_class[space][space]score[space][space]correct_class[space][space]is_correct
def output(filename, data_dict):
    lines = []

    for k, v in data_dict.items():
        line = k

        for i in v:
            line += "  " + str(i)

        lines.append(line)

    lines = "\n".join(lines)

    with open(output_folder / filename, 'w') as file:
        file.writelines(lines)


def run():
    args = get_args()

    train_set = read_data(data_folder / args.train)
    test_set = read_data(data_folder / args.test, col_names=("tweet_id", "text", "q1_label"))

    regular_solution = Naive_Bayes(train_set,test_set, ['yes','no'], False)
    filtered_solution =Naive_Bayes(train_set,test_set, ['yes','no'], True)
    
    print('regular')
    print(regular_solution)
    print('Filtered')
    print(filtered_solution)

    output("trace_NB-BOW-OV.txt", regular_solution)
    output("trace_NB-BOW-FV.txt", filtered_solution)
    
    #train_yes_scores = calc_score(train_set, "yes", False)
    #train_no_scores = calc_score(train_set, "no", False)


    #train_yes_scores_filtered = calc_score(train_set, "yes", True)
    #train_no_scores_filtered = calc_score(train_set, "no", True)

    """#Abstract out this part later once its all figured out


    #Training yes
    train_vocab_yes = build_vocabulary(train_set, label_value="yes")
    train_conditionals_yes = calc_conditionals(train_vocab_yes)
    train_priors_yes = calc_priors(train_set, label_value="yes")


    #Training no
    train_vocab_no = build_vocabulary(train_set, label_value="no")
    train_conditionals_no = calc_conditionals(train_vocab_no)
    train_priors_no = calc_priors(train_set, label_value="no")



    #Same thing, Filtered Edition
    #Training yes
    train_vocab_yes_filtered = build_vocabulary(train_set, label_value="yes", filter_vocab=True)
    train_conditionals_yes_filtered = calc_conditionals(train_vocab_yes_filtered)
    train_priors_yes_filtered = calc_priors(train_set, label_value="yes")


    #Training no
    train_vocab_no_filtered = build_vocabulary(train_set, label_value="no", filter_vocab=True)
    train_conditionals_no_filtered = calc_conditionals(train_vocab_no_filtered)
    train_priors_no_filtered = calc_priors(train_set, label_value="no")"""


    #Now same thing needs to be done for Testing set

    #Testing
    """#test_vocab = build_vocabulary(test_set)
    #test_vocab_filtered = build_vocabulary(test_set, filter_vocab=True)"""


if __name__ == "__main__":
    run()
