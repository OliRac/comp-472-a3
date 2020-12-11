# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:09:49 2020

# By Ihsaan Malek and Olivier Racette
Evaluate model metrics and output to a file

"""

def evaluate(file_name,NB_output,class1,class2):
    
    accuracy = 0
       
    class1_TP = 0       #are class1 and model assigns class1
    class1_FP = 0       #model assigns class 1, but acutally class 2
    class1_FN = 0       #data actual in class 1, but model does not select it
    
    class2_TP = 0
    class2_FP = 0
    class2_FN = 0
    
    #using a confusion matrix to better visualize
    #predicted on rows, correct on columns
    conf_mat = [[0,0],[0,0]]
       
    y = 0   #matrix row
    x = 0   #matrix col

    #building the cofusion matrix values
    for row in NB_output.values():
        #print(row)
        
        #redoing logic
        #if row[0] == class1:
        #    if row[3] == 'correct':
        #        class1_TP += 1
        #        accuracy += 1
        #    else:
                #print('False Positive')
                #class1_FP += 1
                #if its not correct, what is it? false negative or false positive?
        #        if row[3] == class2:    #expecting a no, but got a yes
        #            class1_FP += 1
        #        elif row[3] == class1   #expecting a yes, got a no
        
        #elif row[0] == class2:
        #    if row[3] == 'correct':
                #print('False Positive')
        #        class2_TP += 1
        #        accuracy += 1
        #    else:
                #class2_FP += 1
                #if its not correct, what is it? false negative or false positive?
        
        #false negative
        #if row[0] == class2 and row[3] == class1:
        #    class1_FN +=1
            
        #if row[0] == class1 and row[3] == class2:
        #    class2_FN +=1
    


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

    print(conf_mat)

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
    
    lines.append(str(round(accuracy, 3)))
    precision_str = str(round(class1_precision))+"  "+ str(round(class2_precision))
    lines.append(precision_str)
    
    recall_str = str(round(class1_recall))+"  "+ str(round(class2_recall))
    lines.append(recall_str)
    
    f1_str = str(round(class1_f1))+"  "+ str(round(class2_f1))
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