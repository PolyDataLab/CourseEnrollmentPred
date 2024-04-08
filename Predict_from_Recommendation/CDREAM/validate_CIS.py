import time
import random
import math
import pickle
import torch
import numpy as np
import pandas as pd
from config import Config
import data_helpers as dh
import dataprocess as dp
import tensorflow as tf
from dataprocess import *
from utils import *
from offered_courses import *
import utils

logger = dh.logger_fn("torch-log", "logs/test-{0}.log".format(time.asctime()))

#MODEL = input("☛ Please input the model file you want to test: ")
#MODEL = "./Others/DREAM_2/runs/1663182568/model-09-0.8750-0.3290.model"
#MODEL = "1681419484"
#MODEL = "1681749277"
#MODEL = "1683234575"
MODEL = "1683744082"
#MODEL = './Course_Beacon/runs/1674078249'

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")

MODEL_DIR = dh.load_model_file(MODEL)
#MODEL_DIR = "./Others/DREAM/runs/1663182568/model-09-0.8750-0.3290.model"

def recall_cal(positives, pred_items):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        return float(correct_preds/actual_bsize)
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))

def precision_cal(positives, pred_items):
        #p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        #actual_bsize= p_length
        number_of_rec = len(pred_items)
        if number_of_rec==0: return 0
        return float(correct_preds/number_of_rec)

def f1_score_cal(prec, rec):
        #p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        # correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        #actual_bsize= p_length
        #number_of_rec = len(pred_items)
        nom= 2 * prec * rec
        denom = prec + rec
        if denom==0: return 0
        return float(nom/ denom)

def course_CIS_dept(basket):
    list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
    basket1 = []
    for course in basket:
        flag = 0
        for term in list_of_terms:
            if course.find(term)!= -1:
                flag = 1
        if(flag==1):
            basket1.append(course)
    return basket1   

def course_CIS_dept_filtering(course):
    list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
    flag = 0
    for term in list_of_terms:
        if course.find(term)!= -1:
            flag = 1
    return flag  

def valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, output_path):
    f = open(output_path, "w") #generating text file with recommendation using filtering function
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    #test_data = dh.load_data(Config().TRAININGSET_DIR)
    valid_data = dh.load_data('./Others/DREAM/valid_sample_without_target.json')

    logger.info("✔︎ Test data processing...")
    #test_target = dh.load_data(Config().TESTSET_DIR)
    valid_target = dh.load_data('./Others/DREAM/validation_target_set.json')

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    dr_model = torch.load(MODEL_DIR)

    dr_model.eval()

    item_embedding = dr_model.encode.weight
    hidden = dr_model.init_hidden(Config().batch_size)

    hitratio_numer = 0
    hitratio_denom = 0
    #ndcg = 0.0
    recall = 0.0
    recall_2= 0.0
    precision = 0.0
    precision_2= 0.0
    f1_score = 0.0
    f1_score_2 = 0.0
    precision_temp =0.0
    f1_score_temp =0.0
    #recall_3= 0.0
    count=0
    #test_recall = 0.0
    #last_batch_actual_size = len(valid_data) % Config().batch_size
    for i, x in enumerate(dh.batch_iter(valid_data, Config().batch_size, Config().seq_len, shuffle=False)):
        uids, baskets, lens, prev_idx = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            scores = []
            du_latest = du[l - 1].unsqueeze(0)
            user_baskets = valid_data[valid_data['userID'] == uid].baskets.values[0]
            #print("user_baskets: ", user_baskets)
            item_list1= []
            # calculating <u,p> score for all test items <u,p> pair
            positives = valid_target[valid_target['userID'] == uid].baskets.values[0]  # list dim 1
            target_semester = valid_target[valid_target['userID'] == uid].last_semester.values[0]
            #print("uid: ", uid, " ",positives)
            for x1 in positives:
                item_list1.append(x1)
            #print(positives)

            p_length = len(positives)
            positives2 = torch.LongTensor(positives)
            #print(positives)
            # Deal with positives samples
            scores_pos = list(torch.mm(du_latest, item_embedding[positives2].t()).data.numpy()[0])
            for s in scores_pos:
                scores.append(s)

            # Deal with negative samples
            #negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
            negtives = list(neg_samples[uid])
            for x2 in negtives:
                item_list1.append(x2)
            negtives2 = torch.LongTensor(negtives)
            scores_neg = list(torch.mm(du_latest, item_embedding[negtives2].t()).data.numpy()[0])
            for s in scores_neg:
                scores.append(s)
            #print(item_list1)
            #print(scores)
            # Calculate hit-ratio
            index_k = []
            #top_k1= Config().top_k
            top_k1 = len(positives)
            #print(index_k)
                #print(pred_items)
            f.write("UserID: ")
            # f.write(str(reversed_user_dict[reversed_user_dict3[uid]])+ "| ")
            f.write(str(reversed_user_dict2[uid])+ "| ")
            #f.write("target basket: ")
            # target_courses = []
            # for item2 in positives:
            #     #f.write(str(reversed_item_dict[item2])+ " ")
            #     target_courses.append(reversed_item_dict[item2])
            # target_courses_CIS = course_CIS_dept(target_courses)
            # for item2 in target_courses_CIS:
            #     f.write(item2+ " ")
            #pred_courses_CIS = course_CIS_dept(pred_courses)
            #top_k1 = len(target_courses_CIS)

            #calculate recall
            #recall_2+= recall_cal(positives, index_k)
                    
            #print(offered_courses[l+1])
            if t_idx==1: # we are not cosnidering randomly selected instances for last batch
                k=0
                pred_items= []
                count1= 0
                while(k<top_k1):
                    index = scores.index(max(scores))
                    item1 = item_list1[index]
                    if not utils.filtering(item1, user_baskets, offered_courses[target_semester], item_dict):
                        #if index not in index_k:
                        #if course_CIS_dept_filtering(reversed_item_dict[item1])==1: #only recommend CIS departmental courses
                        if item1 not in pred_items:
                            #index_k.append(index)
                            pred_items.append(item1)
                            k+=1
                    scores[index] = -9999
                    count1+= 1
                    if(count1==len(scores)): break
                
                target_courses = []
                for item2 in positives:
                    f.write(str(reversed_item_dict[item2])+ " ")
                    target_courses.append(reversed_item_dict[item2])
                target_courses_CIS = course_CIS_dept(target_courses)


                f.write(", Recommended basket: ")
                pred_courses = []
                for item3 in pred_items:
                    f.write(str(reversed_item_dict[item3])+ " ")
                    pred_courses.append(reversed_item_dict[item3])


                f.write("\n") 
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer += len((set(positives) & set(pred_items)))
                hitratio_denom += p_length
                #print(index_k)
                # target_courses_CIS = course_CIS_dept(target_courses)
                pred_courses_CIS = course_CIS_dept(pred_courses)

                #calculate recall
                #recall_2+= recall_cal(positives, index_k)
                if len(target_courses_CIS)>0:
                    recall_temp = recall_cal(target_courses_CIS, pred_courses_CIS)
                    recall_2+= recall_temp
                    precision_temp = precision_cal(target_courses_CIS, pred_courses_CIS)
                    precision_2 += precision_temp
                    f1_score_temp = f1_score_cal(recall_temp, precision_temp)
                    f1_score_2 += f1_score_temp
                    count=count+1

    hitratio = hitratio_numer / hitratio_denom
    #ndcg = ndcg / len(test_data)
    recall = recall_2/ count
    precision = precision_2/ count
    f1_score = f1_score_2/ count
    # print('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio))
    # f.write(str('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio)))
    print(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write("\n")
    #print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    print('Recall[@n]: {0}'.format(recall))
    f.write(str('Recall[@n]: {0}'.format(recall)))
    
    print('Precision[@n]: {0}'.format(precision))
    f.write(str('Precision[@n]: {0}'.format(precision)))
    print('F1_score[@n]: {0}'.format(f1_score))
    f.write(str('F1_score[@n]: {0}'.format(f1_score)))
    f.write("\n")

    f.close()


if __name__ == '__main__':
    train_data = pd.read_json('./Filtered_data/train_sample_augmented.json', orient='records', lines= True)
    # train_all, train_set_without_target, target, item_dict, user_dict, reversed_item_dict, reversed_user_dict, max_len = preprocess_train_data(train_data)
    train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data) 
    valid_data = pd.read_json('./valid_data_all.json', orient='records', lines= True)
    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    valid_all = pd.read_json('./Others/DREAM/valid_sample_all.json', orient='records', lines=True)
    valid_set_without_target= pd.read_json('./Others/DREAM/valid_sample_without_target.json', orient='records', lines=True)
    valid_target = pd.read_json('./Others/DREAM/validation_target_set.json', orient='records', lines=True)
    #offered_courses = calculate_offered_courses(valid_all)
    offered_courses = offered_course_cal('./all_data.csv')
    data_dir= './Others/DREAM/'
    output_dir = data_dir + "/output_dir"
    create_folder(output_dir)
    output_path= output_dir+ "/valid_prediction.txt"
    valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, output_path)

