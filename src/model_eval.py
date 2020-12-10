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
    
       
    for row in NB_output.values():
        print(row)
        
        if row[0] == class1:
            if row[3] == 'correct':
                class1_TP += 1
                accuracy += 1
            else:
                print('False Positive')
                class1_FP
        
        elif row[0] == class2:
            if row[3] == 'correct':
                print('False Positive')
                class2_TP += 1
                accuracy += 1
            else:
                class2_FP
        
        #false negative
        if row[0] == class2 and row[3] == class1:
            class1_FN +=1
            
        if row[0] == class1 and row[3] == class2:
            class2_FN +=1
    
    
    class1_precision = class1_TP/(class1_TP + class1_FP)    #TP/(TP+FP)
    class2_precision = class2_TP/(class2_TP + class2_FP)
    
    class1_recall = class1_TP/(class1_TP + class1_FN)      #TP/(TP+FN)
    class2_recall = class2_TP/(class2_TP + class2_FN)
        
    class1_f1 = 2*class1_precision*class1_recall /(class1_precision + class1_recall)
    class2_f1 = 2*class2_precision*class2_recall /(class2_precision + class2_recall)
            
    accuracy = accuracy/len(NB_output)
    
    lines = []
    
    lines.append(round(accuracy, 3))
    precision_str = str(round(class1_precision))+"  "+ str(round(class2_precision))
    lines.append(precision_str)
    
    recall_str = str(round(class1_recall))+"  "+ str(round(class2_recall))
    lines.append(recall_str)
    
    f1_str = str(round(class1_f1))+"  "+ str(round(class2_f1))
    lines.append(f1_str)
    
    with open(file_name, 'w') as file:
        file.writelines(lines)
        
    print('Finish writing Eval file')
    
    