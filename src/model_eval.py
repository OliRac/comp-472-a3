# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:09:49 2020

# By Ihsaan Malek and Olivier Racette
Evaluate model metrics and output to a file

"""
#Calculates the evaluation metrics (accuracy, precision, recall, f1) and outputs it to a file
def evaluate(file_name,NB_output,class1,class2):  
    #Not very generic but hey it works
    #using a confusion matrix to better visualize
    #predicted on rows, correct on columns
    conf_mat = [[0,0],[0,0]]
       
    y = 0   #matrix row
    x = 0   #matrix col

    accuracy = 0

    #building the cofusion matrix values
    for row in NB_output.values():
        if row[3] == 'correct':
            accuracy += 1

            if row[0] == class1:
                y = 0
                x = 0
            if row[0] == class2:
                y = 1
                x = 1
        elif row[2] == class1 and row[0] == class2:    #expected yes but got no
            y = 1
            x = 0
        elif row[2] == class2 and row[0] == class1:     #expected no but got yes
            y = 0
            x = 1

        conf_mat[y][x] += 1

    #TP = diag -> [0][0] and [1][1]
    #FP = sum of row (without diag)
    #FN = sum of col (without diag)

    class1_TP = conf_mat[0][0]
    class2_TP = conf_mat[1][1]

    class1_FP = conf_mat[0][1]
    class1_FN = conf_mat[1][0]

    class2_FP = conf_mat[1][0]  #could also say = class1_FN
    class2_FN = conf_mat[0][1]  #could also say = class1_FP

    class1_precision = class1_TP/(class1_TP + class1_FP)    #TP/(TP+FP)
    class2_precision = class2_TP/(class2_TP + class2_FP)
    
    class1_recall = class1_TP/(class1_TP + class1_FN)      #TP/(TP+FN)
    class2_recall = class2_TP/(class2_TP + class2_FN)
    
    class1_f1 = 2*(class1_precision*class1_recall) /(class1_precision + class1_recall)
    class2_f1 = 2*(class2_precision*class2_recall) /(class2_precision + class2_recall)
            
    accuracy = accuracy/len(NB_output)
    
    lines = []
    
    num_digits = 3

    lines.append(str(round(accuracy, num_digits)))
    precision_str = str(round(class1_precision, num_digits))+"  "+ str(round(class2_precision, num_digits))
    lines.append(precision_str)
    
    recall_str = str(round(class1_recall, num_digits))+"  "+ str(round(class2_recall, num_digits))
    lines.append(recall_str)
    
    f1_str = str(round(class1_f1, num_digits))+"  "+ str(round(class2_f1, num_digits))
    lines.append(f1_str)

    lines = "\n".join(lines)

    with open(file_name, 'w') as file:
        file.writelines(lines)
        
    print("Finished writing evaluation data to " + str(file_name))  


#Outputs the result data dictionary to a file in the output directory. 
#Uses the following format: tweet_id[space][space]predicted_class[space][space]score[space][space]correct_class[space][space]is_correct
def output_trace(file_path, data_dict):
    lines = []

    for k, v in data_dict.items():
        line = k

        for i in v:
            line += "  " + str(i)

        lines.append(line)

    lines = "\n".join(lines)

    with open(file_path, 'w') as file:
        file.writelines(lines)

    print("Finished writing trace data to " + str(file_path))