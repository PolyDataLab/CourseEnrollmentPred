import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import random
import tqdm
from preprocess import *
from utils import *
from offered_courses import *
from preprocess import *
from sklearn.cluster import KMeans
import numpy as np
import math
from keras import models, layers


#This function counts the number of apperances of a course in a particular semester and store them in the dictionary
#Semester_dict consists of all possible semester numbers as keys and a dictionary of count of courses as values
def calculate_dict(semester_dict, index2, basket):
    if index2 in semester_dict:
        count_item= semester_dict[index2]
    else:
        count_item = {}
    for item2 in basket:
        count_item[item2]= count_item.get(item2, 0)+ 1
    semester_dict[index2] = count_item
    return semester_dict

#training process to measure popular courses
#it returns a dictionary of count of courses

#recommend top-k courses based on highest score of courses
def recommend_top_k(main_dict, ts, user_baskets, offered_course_list, top_k, item_list):
     top_k1= 0
     #print(prob_dict)
     top_items= []
     for keys, values in main_dict.items():
            if(keys==ts):
                count_dict= values
                top_k1=0
                for item in count_dict.keys():
                    if not filtering(item, user_baskets, offered_course_list, item_list):
                        top_items.append(item)
                        top_k1+=1
                        if(top_k1==top_k): break
     return top_items

#calculate recall 
def recall_cal(top_item_list, target_item_list, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred):
    t_length= len(target_item_list)
    correct_preds= len((set(top_item_list) & set(target_item_list)))
    #print(correct_preds)
    actual_bsize= t_length
    if correct_preds>=1: count_at_least_one_cor_pred+= 1
    if correct_preds>=2: count_at_least_two_cor_pred+= 1
    if correct_preds>=3: count_at_least_three_cor_pred+= 1
    if correct_preds>=4: count_at_least_four_cor_pred+= 1
    if correct_preds>=5: count_at_least_five_cor_pred+= 1
    if correct_preds==actual_bsize: count_all_cor_pred+= 1

    if (actual_bsize>=6): 
        if(correct_preds==1): count_cor_pred[6,1]+= 1
        if(correct_preds==2): count_cor_pred[6,2]+= 1
        if(correct_preds==3): count_cor_pred[6,3]+= 1
        if(correct_preds==4): count_cor_pred[6,4]+= 1
        if(correct_preds==5): count_cor_pred[6,5]+= 1
        if(correct_preds>=6): count_cor_pred[6,6]+= 1
    else:
        if(correct_preds==1): count_cor_pred[actual_bsize,1]+= 1
        if(correct_preds==2): count_cor_pred[actual_bsize,2]+= 1
        if(correct_preds==3): count_cor_pred[actual_bsize,3]+= 1
        if(correct_preds==4): count_cor_pred[actual_bsize,4]+= 1
        if(correct_preds==5): count_cor_pred[actual_bsize,5]+= 1

    return float(correct_preds/actual_bsize), count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred
# def course_CIS_dept(basket):
#     list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
#     basket1 = []
#     for course in basket:
#         flag = 0
#         for term in list_of_terms:
#             if course.find(term)!= -1:
#                 flag = 1
#         if(flag==1):
#             basket1.append(course)
#     return basket1  


# calculating term dictionary where key = semester and value = another dictionary with course and number of actual enrolments
def calculate_term_dict(term_dict, semester, basket, reversed_item_dict):
    for item in basket:
        if semester not in term_dict:
            count_course = {}
        else:
            count_course = term_dict[semester]
        if reversed_item_dict[item] not in count_course:
            count_course[reversed_item_dict[item]] = 1
        else:
            count_course[reversed_item_dict[item]] = count_course[reversed_item_dict[item]]+ 1
        term_dict[semester] = count_course
    return term_dict

def calculate_term_dict_2(term_dict, semester, basket):
    for item in basket:
        if semester not in term_dict:
            count_course = {}
        else:
            count_course = term_dict[semester]
        if item not in count_course:
            count_course[item] = 1
        else:
            count_course[item] = count_course[item] + 1
        
        term_dict[semester] = count_course
    return term_dict

def calculate_term_dict_true(term_dict_true, semester, t_basket, pred_basket):
    for item in pred_basket:
        if item in t_basket:
            if semester not in term_dict_true:
                count_course = {}
            else:
                count_course = term_dict_true[semester]
            if item not in count_course:
                count_course[item] = 1
            else:
                count_course[item] = count_course[item] + 1
            term_dict_true[semester] = count_course
    return term_dict_true

def calculate_term_dict_false(term_dict_false, semester, t_basket, pred_basket):
    for item in pred_basket:
        if item not in t_basket:
            if semester not in term_dict_false:
                count_course = {}
            else:
                count_course = term_dict_false[semester]
            if item not in count_course:
                count_course[item] = 1
            else:
                count_course[item] = count_course[item]+ 1
            term_dict_false[semester] = count_course
    return term_dict_false

def calculate_avg_n_actual_courses(input_data):
    data = input_data
    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if item not in frequency_of_courses:
                    frequency_of_courses[item] = 1
                else:
                    frequency_of_courses[item] += 1
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict_2(term_dict_all, semester, basket)
    count_course_all = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in count_course_all:
                count_course_all[item] = [cnt, 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = count_course_all[item]
                cnt1 += cnt
                n1 += 1
                #count_course_all[item] = list1
                count_course_all[item] = [cnt1, n1]
    count_course_avg = {}
    for course, n in count_course_all.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        count_course_avg[course] = float(cnt2/n2)
    #calculate standard deviation
    course_sd = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in course_sd:
                course_sd[item] = [pow((cnt-count_course_avg[item]),2), 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = course_sd[item]
                cnt1 = cnt1+ pow((cnt-count_course_avg[item]),2)
                n1 += 1
                #count_course_all[item] = list1
                course_sd[item] = [cnt1, n1]
    course_sd_main = {}
    course_number_terms = {}
    for course, n in course_sd.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        if(n2==1): course_sd_main[course] = float(math.sqrt(cnt2/n2))
        else: course_sd_main[course] = float(math.sqrt(cnt2/(n2-1)))
        course_number_terms[course] = n2
    
    return term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms

def find_prior_term(course, prior_semester, term_dict_all_prior):
    flag = 0
    count_course_prior_2 = {}
    while(flag!=1):
        #print("prior_semester: ", prior_semester)
        if prior_semester in term_dict_all_prior:
            count_course_prior_2 = term_dict_all_prior[prior_semester]
        if course in count_course_prior_2:
            flag =1
        if prior_semester %5==0:
            prior_semester = prior_semester-4
        else:
            prior_semester = prior_semester-3
    return count_course_prior_2 

# checking if a course is a CIS course or not by using the list of acronyms for CIS courses
def course_CIS_dept_filtering(course):
    list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
    flag = 0
    for term in list_of_terms:
        if course.find(term)!= -1:
            flag = 1
    return flag

def calculate_std_dev(error_list):
    sum_err = 0.0
    for err in error_list:
        sum_err += err
    avg_err = sum_err/ len(error_list)
    sum_diff = 0.0
    for err in error_list:
        sum_diff += pow((err-avg_err), 2)
    std_dev = math.sqrt((sum_diff/len(error_list)))
    return avg_err, std_dev

def prepare_train_data(train_data, train_courses, avg_enrollment_train, reversed_item_dict):
    #course_list = []
    terms = []
    for col in avg_enrollment_train.columns:
        terms.append(col)
    avg_en_dict = {}
    for x in range(len(avg_enrollment_train)):
        for y in terms:
            avg_en_dict[y] = avg_enrollment_train[y][x]

    index1 = 0
    courses_term_1201 =[]
    courses_term_1198 =[]
    courses_term_1195 =[]
    for baskets in train_data['baskets']:
        ts = train_data['timestamps'][index1]
        index2, index3, index4 = -1, -1, -1
        for t in range(len(ts)):
            if(ts[t]==1201): index2= t
            if(ts[t]==1198): index3= t
            if(ts[t]==1195): index4= t
        if index2!=-1:
            basket = baskets[index2]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1201:
                    courses_term_1201.append(reversed_item_dict[item])
        if index3!=-1:
            basket = baskets[index3]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1198:
                    courses_term_1198.append(reversed_item_dict[item])
        if index4!=-1:
            basket = baskets[index4]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1195:
                    courses_term_1195.append(reversed_item_dict[item])
        index1 += 1
    print(courses_term_1195)
    train_data_en_all = []
    for course in courses_term_1201:
        train_data_en = []
        for courses in train_courses['course_id']:
            if course==courses:
                index = train_courses[train_courses['course_id'] == course].index.values[0]
                #prior_terms = ['1198', '1195', '1191', '1188', '1185', '1181', '1178', '1175', '1171', '1168', '1165', '1161', '1158', '1155', '1151']
                #prior_terms = ['1151', '1155', '1158','1161', '1165', '1168', '1171', '1175', '1178', '1181', '1185', '1188', '1191', '1195', '1198']
                prior_terms = ['1171', '1175', '1178', '1181', '1185', '1188', '1191', '1195', '1198']
                for pt in prior_terms:
                    if train_courses[pt][index]!=0:
                        train_data_en.append(train_courses[pt][index])
                    else:
                        train_data_en.append(avg_en_dict[int(pt)])
                # train_data_en.append(train_courses['level1'][index])
                # train_data_en.append(train_courses['level2'][index])
                # train_data_en.append(train_courses['level3'][index])
                # train_data_en.append(train_courses['level4'][index])
                # train_data_en.append(train_courses['level5'][index])
                # train_data_en.append(avg_en_dict[1198])
                train_data_en.append(train_courses['1201'][index])
                train_data_en_all.append(train_data_en)
    
    for course in courses_term_1198:
        train_data_en = []
        for courses in train_courses['course_id']:
            if course==courses:
                index = train_courses[train_courses['course_id'] == course].index.values[0]
                #prior_terms = ['1195', '1191', '1188', '1185', '1181', '1178', '1175', '1171', '1168', '1165', '1161', '1158', '1155', '1151', '1148']
                #prior_terms = ['1148', '1151', '1155', '1158','1161', '1165', '1168', '1171', '1175', '1178', '1181', '1185', '1188', '1191', '1195']
                prior_terms = ['1168', '1171', '1175', '1178', '1181', '1185', '1188', '1191', '1195']
                for pt in prior_terms:
                    if train_courses[pt][index]!=0:
                        train_data_en.append(train_courses[pt][index])
                    else:
                        train_data_en.append(avg_en_dict[int(pt)])
                # train_data_en.append(train_courses['level1'][index])
                # train_data_en.append(train_courses['level2'][index])
                # train_data_en.append(train_courses['level3'][index])
                # train_data_en.append(train_courses['level4'][index])
                # train_data_en.append(train_courses['level5'][index])
                # train_data_en.append(avg_en_dict[1195])
                train_data_en.append(train_courses['1198'][index])
                train_data_en_all.append(train_data_en)
    
    for course in courses_term_1195:
        train_data_en = []
        for courses in train_courses['course_id']:
            if course==courses:
                index = train_courses[train_courses['course_id'] == course].index.values[0]
                #prior_terms = ['1191', '1188', '1185', '1181', '1178', '1175', '1171', '1168', '1165', '1161', '1158', '1155', '1151', '1148', '1145']
                #prior_terms = ['1145', '1148', '1151', '1155', '1158','1161', '1165', '1168', '1171', '1175', '1178', '1181', '1185', '1188', '1191']
                prior_terms = ['1165', '1168', '1171', '1175', '1178', '1181', '1185', '1188', '1191']
                for pt in prior_terms:
                    if train_courses[pt][index]!=0:
                        train_data_en.append(train_courses[pt][index])
                    else:
                        train_data_en.append(avg_en_dict[int(pt)])
                # train_data_en.append(train_courses['level1'][index])
                # train_data_en.append(train_courses['level2'][index])
                # train_data_en.append(train_courses['level3'][index])
                # train_data_en.append(train_courses['level4'][index])
                # train_data_en.append(train_courses['level5'][index])
                # train_data_en.append(avg_en_dict[1191])
                train_data_en.append(train_courses['1195'][index])
                train_data_en_all.append(train_data_en)
    #train_data_en_all1 = pd.DataFrame(train_data_en_all, columns=['n_en_t1', 'n_en_t2', 'n_en_t3', 'n_en_t4', 'n_en_t5', 'n_en_t6', 'n_en_t7', 'n_en_t8', 'n_en_t9', 'n_en_t10', 'n_en_t11', 'n_en_t12', 'n_en_t13', 'n_en_t14', 'n_en_t15', 'target_enr_n'])
    train_data_en_all1 = pd.DataFrame(train_data_en_all, columns=['n_en_t1', 'n_en_t2', 'n_en_t3', 'n_en_t4', 'n_en_t5', 'n_en_t6', 'n_en_t7', 'n_en_t8', 'n_en_t9', 'target_enr_n'])
    train_data_en_all1.to_csv('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/train_data_en_all2.csv')
    train_data_en_all1.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/train_data_en_all2.json', orient='records', lines=True)
    return train_data_en_all1

# def prepare_train_data_lag(train_data):
#     nLags = 3
#     mu = 0.000001
#     data = train_data
#     cols = []
#     for col in data.columns:
#         cols.append(col)
#     course_data_all = pd.DataFrame()
#     Data_Lags_all = pd.DataFrame()
#     for x in range(len(data)):
#         course_data1 = []
#         for col2 in cols:
#             course_data1.append(data[col2][x])
#         course_data = pd.DataFrame(course_data1)
#         Data_Lags = pd.DataFrame(np.zeros((len(course_data), nLags)))
#         for i in range(0, nLags):
#             Data_Lags[i] = course_data.shift(i + 1)
#         Data_Lags = Data_Lags[nLags:]
#         course_data = course_data[nLags:]
#         Data_Lags.index = np.arange(0, len(Data_Lags), 1, dtype=int)
#         course_data.index = np.arange(0, len(course_data), 1, dtype=int)
#         #print("new line 1")
#         #print(Data_Lags)
#         #print("new line 2")
#         #print(course_data)
#         course_data_all = pd.concat([course_data_all, course_data], ignore_index=True)
#         Data_Lags_all = pd.concat([Data_Lags_all, Data_Lags], ignore_index=True)

#         #if x==0: break
#     #train_size = int(len(data) * 0.8)
    
#     return course_data_all, Data_Lags_all

def prepare_train_data_seq(train_data):
    nLags = 3
    mu = 0.000001
    data = train_data
    cols = []
    for col in data.columns:
        cols.append(col)
    #course_data_all = pd.DataFrame()
    #Data_Lags_all = pd.DataFrame()

    x_all = []
    y = []
    seq_size = nLags
    #dataset = dataset.reshape(-1, 1)
    dataset = data
    for k in range(len(dataset)):
        x = []
        for k1 in range(0, (len(cols)-1), seq_size):
            #window = dataset[k1:(k1 + seq_size), 0][k]
            window = [dataset[cols[k1]][k], dataset[cols[k1+1]][k], dataset[cols[k1+2]][k]]
            x.append(window)
            #k1 += 3
        x_all.append(x)
        y.append(dataset[cols[-1]][k])
        
    #print(x_all)
    x_all = np.reshape(np.array(x_all), (np.array(x_all).shape[0], np.array(x_all).shape[1], 3))  # Reshape input to be [samples, time steps, features]
    print(x_all)
    return x_all, np.array(y)

def prepare_test_data_seq(test_data):
    nLags = 3
    mu = 0.000001
    data = test_data
    cols = []
    for col in data.columns:
        cols.append(col)
    #course_data_all = pd.DataFrame()
    #Data_Lags_all = pd.DataFrame()

    x_all = []
    y = []
    seq_size = nLags
    #dataset = dataset.reshape(-1, 1)
    dataset = data
    for k in range(len(dataset)):
        x = []
        for k1 in range(0, (len(cols)-1), seq_size):
            #window = dataset[k1:(k1 + seq_size), 0][k]
            window = [dataset[cols[k1]][k], dataset[cols[k1+1]][k], dataset[cols[k1+2]][k]]
            x.append(window)
            #k1 += 3
        x_all.append(x)
        y.append(dataset[cols[-1]][k])
        
    #print(x_all)
    x_all = np.reshape(np.array(x_all), (np.array(x_all).shape[0], np.array(x_all).shape[1], 3))  # Reshape input to be [samples, time steps, features]
    print(x_all)
    return x_all, np.array(y)



def prepare_test_data(test_target, all_courses, avg_enrollment_all, term_dict_test):
    #course_list = []
    terms = []
    for col in avg_enrollment_all.columns:
        terms.append(col)
    avg_en_dict = {}
    for x in range(len(avg_enrollment_all)):
        for y in terms:
            avg_en_dict[y] = avg_enrollment_all[y][x]

    index1 = 0
    courses_term_1221 =[]
    courses_term_1218 =[]
    courses_term_1215 =[]
    for basket in test_target['baskets']:
        ls = test_target['last_semester'][index1] 
        if ls==1221:
            for item in basket:
                if item not in courses_term_1221:
                    courses_term_1221.append(item)
        if ls==1218:
            for item in basket:
                if item not in courses_term_1218:
                    courses_term_1218.append(item)
        if ls==1215:
            for item in basket:
                if item not in courses_term_1215:
                    courses_term_1215.append(item)
        index1 += 1
    
    #print(courses_term_1215)

    test_data_en_all = []
    for course in courses_term_1221:
        test_data_en = []
        count_actual_n = term_dict_test[1221]
        for courses in all_courses['course_id']:
            if course==courses and course_CIS_dept_filtering(course)==1: #considering CIS courses only
                index = all_courses[all_courses['course_id'] == course].index.values[0]
                # prior_terms = ['1218', '1215', '1211', '1208', '1205', '1201', '1198', '1195', '1191', '1188', '1185', '1181', '1178', '1175', '1171']
                #prior_terms = ['1171', '1175', '1178', '1181', '1185', '1188','1191', '1195', '1198', '1201', '1205', '1208','1211', '1215', '1218' ]
                prior_terms = ['1191', '1195', '1198', '1201', '1205', '1208','1211', '1215', '1218']
                #prior_terms = ['1218', '1215', '1211']
                for pt in prior_terms:
                    if all_courses[pt][index]!=0:
                        test_data_en.append(all_courses[pt][index])
                    else:
                        test_data_en.append(avg_en_dict[int(pt)])
                # test_data_en.append(all_courses['level1'][index])
                # test_data_en.append(all_courses['level2'][index])
                # test_data_en.append(all_courses['level3'][index])
                # test_data_en.append(all_courses['level4'][index])
                # test_data_en.append(all_courses['level5'][index])
                # test_data_en.append(avg_en_dict[1218])
                test_data_en.append(all_courses['1221'][index])
                #test_data_en.append(count_actual_n[course])
                test_data_en_all.append(test_data_en)
        
    for course in courses_term_1218:
        test_data_en = []
        count_actual_n = term_dict_test[1218]
        for courses in all_courses['course_id']:
            if course==courses and course_CIS_dept_filtering(course)==1:
                index = all_courses[all_courses['course_id'] == course].index.values[0]
                #prior_terms = ['1215', '1211', '1208', '1205', '1201', '1198', '1195', '1191', '1188', '1185', '1181', '1178', '1175', '1171', '1168']
                #prior_terms = ['1215', '1211', '1208']
                #prior_terms = ['1168', '1171', '1175', '1178', '1181', '1185', '1188','1191', '1195', '1198', '1201', '1205', '1208','1211', '1215']
                prior_terms = ['1188','1191', '1195', '1198', '1201', '1205', '1208','1211', '1215']
                for pt in prior_terms:
                    if all_courses[pt][index]!=0:
                        test_data_en.append(all_courses[pt][index])
                    else:
                        test_data_en.append(avg_en_dict[int(pt)])
                # test_data_en.append(all_courses['level1'][index])
                # test_data_en.append(all_courses['level2'][index])
                # test_data_en.append(all_courses['level3'][index])
                # test_data_en.append(all_courses['level4'][index])
                # test_data_en.append(all_courses['level5'][index])
                # test_data_en.append(avg_en_dict[1215])
                test_data_en.append(all_courses['1218'][index])
                #test_data_en.append(count_actual_n[course])
                test_data_en_all.append(test_data_en)
    
    for course in courses_term_1215:
        test_data_en = []
        count_actual_n = term_dict_test[1215]
        for courses in all_courses['course_id']:
            if course==courses and course_CIS_dept_filtering(course)==1: # considering CIS courses only
                index = all_courses[all_courses['course_id'] == course].index.values[0]
                #prior_terms = ['1211', '1208', '1205', '1201', '1198', '1195', '1191', '1188', '1185', '1181', '1178', '1175', '1171', '1168', '1165']
                #prior_terms = ['1211', '1208', '1205']
                # prior_terms = ['1165', '1168', '1171', '1175', '1178', '1181', '1185', '1188','1191', '1195', '1198', '1201', '1205', '1208','1211']
                prior_terms = ['1185', '1188','1191', '1195', '1198', '1201', '1205', '1208','1211']
                for pt in prior_terms:
                    if all_courses[pt][index]!=0:
                        test_data_en.append(all_courses[pt][index])
                    else:
                        test_data_en.append(avg_en_dict[int(pt)])
                # test_data_en.append(all_courses['level1'][index])
                # test_data_en.append(all_courses['level2'][index])
                # test_data_en.append(all_courses['level3'][index])
                # test_data_en.append(all_courses['level4'][index])
                # test_data_en.append(all_courses['level5'][index])
                # test_data_en.append(avg_en_dict[1211])
                test_data_en.append(all_courses['1215'][index])
                #test_data_en.append(count_actual_n[course])
                test_data_en_all.append(test_data_en)
    
    
    # test_data_en_all1 = pd.DataFrame(test_data_en_all, columns=['n_en_t1', 'n_en_t2', 'n_en_t3', 'n_en_t4', 'level1', 'level2', 'level3', 'level4', 'level5', 'avg_n_prior', 'target_enr_n'])
    #test_data_en_all1 = pd.DataFrame(test_data_en_all, columns=['n_en_t1', 'n_en_t2', 'n_en_t3', 'n_en_t4', 'n_en_t5', 'n_en_t6', 'n_en_t7', 'n_en_t8', 'n_en_t9', 'n_en_t10', 'n_en_t11', 'n_en_t12', 'n_en_t13', 'n_en_t14', 'n_en_t15', 'target_enr_n'])
    test_data_en_all1 = pd.DataFrame(test_data_en_all, columns=['n_en_t1', 'n_en_t2', 'n_en_t3', 'n_en_t4', 'n_en_t5', 'n_en_t6', 'n_en_t7', 'n_en_t8', 'n_en_t9', 'target_enr_n'])
    #test_data_en_all1 = pd.DataFrame(test_data_en_all, columns=['n_en_t1', 'n_en_t2', 'n_en_t3', 'target_enr_n'])
    test_data_en_all1.to_csv('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/test_data_en_all2_LSTM_CIS.csv')
    test_data_en_all1.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/test_data_en_all2_LSTM_CIS.json', orient='records', lines=True)
    return test_data_en_all1

# LSTM for regression model
def Sequence_model(train_x, train_y, test_x, test_y):
    mod = models.Sequential()  # Build the model
    # mod.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), activation='relu', input_shape=(None, nLags)))  # ConvLSTM2D
    # mod.add(layers.Flatten())
    nLags = 3
    mod.add(layers.LSTM(units=128, activation='tanh', input_shape=(None, nLags)))
    mod.add(layers.Dropout(rate=0.2))
    # mod.add(layers.LSTM(units=100, activation='tanh'))  # Stacked LSTM
    # mod.add(layers.Bidirectional(layers.LSTM(units=100, activation='tanh'), input_shape=(None, 1)))     # Bidirectional LSTM: forward and backward
    mod.add(layers.Dense(64))
    mod.add(layers.Dense(1))   # A Dense layer of 1 node is added in order to predict the label(Prediction of the next value)
    mod.compile(optimizer='adam', loss='mae')
    mod.fit(train_x, train_y, validation_data=(test_x, test_y), verbose=2, epochs=200)

    #y_train_pred = pd.Series(mod.predict(train_x).ravel())
    return mod

def evaluate_test_data_new(test_x, test_y, regr):
    # X = test_data_en_all.iloc[:, 0:-1]
    # y = test_data_en_all.iloc[:, -1]
    #en_pred = regr.predict(X)
    en_pred = pd.Series(regr.predict(test_x).ravel())
    y= test_y

    course_en_ac_pr = []
    for x in range(len(y)):
        row = [y[x], en_pred[x]]
        course_en_ac_pr.append(row)
    
    course_en_ac_pr = pd.DataFrame(course_en_ac_pr, columns=['actual_enrollment', 'predicted_enrollment'])
    course_en_ac_pr.to_csv('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/course_en_ac_pr_2_LSTM_R_CIS_v5.csv')
    course_en_ac_pr.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/course_en_ac_pr_2_LSTM_CIS_v5.json', orient='records', lines=True)

    return y, en_pred, course_en_ac_pr

def calculate_mse_for_course_allocation(actual_en, en_pred):
    mse_for_course_allocation = 0.0
    mse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mse_for_course_allocation_considering_not_predicted_courses = 0.0
    mae_for_course_allocation = 0.0
    mae_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mae_for_course_allocation_considering_not_predicted_courses = 0.0
    msse_for_course_allocation = 0.0
    msse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_msse_for_course_allocation_considering_not_predicted_courses = 0.0
    mase_for_course_allocation = 0.0
    mase_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mase_for_course_allocation_considering_not_predicted_courses = 0.0
    #count1= 0
    count2 = 0
    output_path1=  "/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/test_course_allocation_v2_2_LSTM_R_CIS_v5.txt"
    #f = open(output_path1, "w") #generating text file with recommendation using filtering function
    #course_allocation = []
    error_list = []
    ab_error_list = []
    st_error_list = []
    for x in range(len(actual_en)):
        mse_for_course_allocation += pow((en_pred[x]-actual_en[x]), 2)
        mae_for_course_allocation += abs(en_pred[x]-actual_en[x])
        if actual_en[x]!=0:
            msse_for_course_allocation += pow(abs((en_pred[x]-actual_en[x])/actual_en[x]), 2)
            mase_for_course_allocation += abs((en_pred[x]-actual_en[x])/actual_en[x])
        else:
            msse_for_course_allocation += 0
            mase_for_course_allocation += 0
        error_list.append(en_pred[x]-actual_en[x])
        ab_error_list.append(abs(en_pred[x]-actual_en[x]))
        if actual_en[x]!=0:
            st_error_list.append(abs((en_pred[x]-actual_en[x])/actual_en[x]))
        else:
            st_error_list.append(0)
        count2 += 1
    #avg_mse_for_course_allocation = mse_for_course_allocation/ count1
    avg_mse_for_course_allocation = mse_for_course_allocation/ count2
    avg_mae_for_course_allocation = mae_for_course_allocation/ count2
    avg_msse_for_course_allocation = msse_for_course_allocation/ count2
    avg_mase_for_course_allocation = mase_for_course_allocation/ count2
    avg_rmse_for_course_allocation = math.sqrt(avg_mse_for_course_allocation)
    avg_rmsse_for_course_allocation = math.sqrt(avg_msse_for_course_allocation)
    mean_error, std_dev_error = calculate_std_dev(error_list)
    mean_ab_error, std_dev_ab_error = calculate_std_dev(ab_error_list)
    mean_st_error, std_dev_st_error = calculate_std_dev(st_error_list)

    print("avg_mse_for_course_allocation_considering all courses available in test data: ",avg_mse_for_course_allocation)
    print("avg rmse for # of allocated course where we are predicting a course at least once: ",avg_rmse_for_course_allocation)
    print("avg_mae_for_course_allocation_considering all courses available in test data: ",avg_mae_for_course_allocation)
    print("avg_rmse_for_course_allocation_considering all courses available in test data: ",avg_rmse_for_course_allocation)
    print("avg_mase_for_course_allocation_considering all courses available in test data: ",avg_mase_for_course_allocation)
    print("avg_rmsse_for_course_allocation_considering all courses available in test data: ",avg_rmsse_for_course_allocation)
    print("mean of errors: ", mean_error)
    print("standard_deviation for errors: ", std_dev_error)
    print("mean of absolute errors: ", mean_ab_error)
    print("standard_deviation for absolute errors: ", std_dev_ab_error)
    print("mean of normalized errors: ", mean_st_error)
    print("standard_deviation for normalized errors: ", std_dev_st_error)

    #f.close()
    # course_allocation_actual_predicted = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'predicted_n', 'predicted_n_true', 'predicted_n_false', 'avg_n_actual', 'st_dev_actual', 'number_of_terms', 'n_sts_last_offering'])
    # course_allocation_actual_predicted.to_csv('/Users/mkhan149/Downloads/Experiments/Others/Gaussian_processes/test_course_allocation_v2.csv')
    #return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list




if __name__ == '__main__':
#    train, test, valid = split_data('/Users/mdakibzabedkhan/Downloads/Experiments/Others/DREAM_2/train_sample.csv')
#     train, test, valid = split_data('/Users/mkhan149/Downloads/Experiments/Others/Gaussian_processes_2/train_sample.csv')
   train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_en_pred_filtered.json', orient='records', lines= True)
   train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
   train_all, train_set_without_target, target, max_len = preprocess_train_data_part2(train_data) 
   #print(len(item_dict))
#    print(train_all)
#    print("max_len:", max_len)
   #print(target)
   #valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/valid_data_all.json', orient='records', lines= True)
   valid_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_all.json', orient='records', lines= True)
   valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
   valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data) #  #, 
   #print("reversed_user_dict2: ", reversed_user_dict2)
   #print(valid_all)
   test_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_all.json', orient='records', lines= True)
   #test_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/test_data_all.json', orient='records', lines= True)
   test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
   test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data) #, item_dict, user_dict, reversed_item_dict, reversed_user_dict #
   term_dict_test, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(test_all)
   #print(term_dict_test[1221])
   print("step 4 done")
   offered_courses = offered_course_cal('/Users/mkhan149/Downloads/Experiments/all_data.json')
#    train_courses = pd.read_json('/Users/mkhan149/Downloads/Experiments/course_df.json', orient='records', lines= True)
#    avg_enrollment_train = pd.read_json('/Users/mkhan149/Downloads/Experiments/avg_enrollment_prior_term.json', orient='records', lines= True)
   all_courses_en = pd.read_json('/Users/mkhan149/Downloads/Experiments/course_df_all.json', orient='records', lines= True)
   avg_enrollment_all = pd.read_json('/Users/mkhan149/Downloads/Experiments/avg_enrollment_prior_term_all.json', orient='records', lines= True)
   train_data_en_all = prepare_train_data(train_all, all_courses_en, avg_enrollment_all, reversed_item_dict)
#    train_data_new, Data_lags_all = prepare_train_data_lag(train_data_en_all)
   train_x, train_y = prepare_train_data_seq(train_data_en_all)
   #print(train_data_new)

#    regr = Gaussian_processes_main(train_data_en_all)
   #regr = Gaussian_processes_main_2(train_data_new, Data_lags_all)

   test_data_en_all = prepare_test_data(test_target, all_courses_en, avg_enrollment_all, term_dict_test)
   test_x, test_y = prepare_test_data_seq(test_data_en_all)

   regr_model = Sequence_model(train_x, train_y, test_x, test_y)
#    y_test, en_pred, course_en_ac_pred = evaluate_test_data(test_data_en_all, regr_model)
   y_test, en_pred, course_en_ac_pred = evaluate_test_data_new(test_x, test_y, regr_model)
   # calculate mse, rmse, mae
   calculate_mse_for_course_allocation(y_test, en_pred)