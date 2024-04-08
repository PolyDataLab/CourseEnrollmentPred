import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import random
import tqdm
from preprocess import *
from utils import *
from offered_courses import *
from preprocess import *
#from sklearn.cluster import KMeans
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

# calculating term dictionary where key = semester and value = dictionary with course and number of actual enrollments
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
        #if semester==1221 and item=="COP4710": print("Count of course COP4710 in 1221 semester:", count_course[item])
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

# calculate average frequency of courses over semesters
def calculate_avg_n_actual_courses_train(input_data, reversed_item_dict):
    data = input_data
    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if reversed_item_dict[item] not in frequency_of_courses:
                    frequency_of_courses[reversed_item_dict[item]] = 1
                else:
                    frequency_of_courses[reversed_item_dict[item]] += 1
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict(term_dict_all, semester, basket, reversed_item_dict)
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

# standard deviation of errors
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

def find_prior_term3(prior_semester, train_data, term_dict_train):
    if prior_semester %5==0:
        prior_semester = prior_semester-4
    else:
        prior_semester = prior_semester-3
    #courses_term_prior = []
    # index1 =0
    # for baskets in train_data['baskets']:
    #     ts = train_data['timestamps'][index1]
    #     index2 = -1
    #     for t in range(len(ts)):
    #         if(ts[t]==prior_semester): index2= t
    #     if index2!=-1:
    #         basket = baskets[index2]
    #         for item in basket:
    #             if reversed_item_dict[item] not in courses_term_prior:
    #                 courses_term_prior.append(reversed_item_dict[item])
    #     index1 += 1
    count_course_prior_2 = term_dict_train[prior_semester]
    courses_term_prior = list(count_course_prior_2.keys())
    return prior_semester, courses_term_prior
    

def prepare_train_data(train_data, item_list, item_dict, reversed_item_dict, term_dict_train):
    
    index1 = 0
    courses_term_1201 =[]
    courses_term_1198 =[]
    courses_term_1195 =[]
    for baskets in train_data['baskets']:
        ls = train_data['last_semester'][index1] 
        ts = train_data['timestamps'][index1]
        index2, index3, index4 = -1, -1, -1
        for t in range(len(ts)):
            if(ts[t]==1201): index2= t
            if(ts[t]==1198): index3= t
            if(ts[t]==1195): index4= t
        if index2!=-1 and ls==1201:
            basket = baskets[index2]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1201:
                    courses_term_1201.append(reversed_item_dict[item])
        if index3!=-1 and ls==1198:
            basket = baskets[index2]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1198:
                    courses_term_1198.append(reversed_item_dict[item])
        if index4!=-1 and ls==1195:
            basket = baskets[index2]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1195:
                    courses_term_1195.append(reversed_item_dict[item])
        index1 += 1
    
    index1 = 0
    courses_term_1191 =[]
    courses_term_1188 =[]
    courses_term_1185 =[]
    for baskets in train_data['baskets']:
        ls = train_data['last_semester'][index1] 
        ts = train_data['timestamps'][index1]
        index2, index3, index4 = -1, -1, -1
        for t in range(len(ts)):
            if(ts[t]==1191): index2= t
            if(ts[t]==1188): index3= t
            if(ts[t]==1185): index4= t
        if index2!=-1 and ls==1191:
            basket = baskets[index2]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1191:
                    courses_term_1191.append(reversed_item_dict[item])
        if index3!=-1 and ls==1188:
            basket = baskets[index2]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1188:
                    courses_term_1188.append(reversed_item_dict[item])
        if index4!=-1 and ls==1185:
            basket = baskets[index2]
            for item in basket:
                if reversed_item_dict[item] not in courses_term_1185:
                    courses_term_1185.append(reversed_item_dict[item])
        index1 += 1
    #print(courses_term_1195)
    train_data_en_all = []
    course_track = {}
    for course in courses_term_1201:
        train_data_en = []
        index2 =0
        for baskets in train_data['baskets']:
            ts = train_data['timestamps'][index2]
            ls = train_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1201 and ls==1201:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(reversed_item_dict[item2]==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track:
                    course_track[item_dict[course]] = [row]
                else:
                    course_track[item_dict[course]].append(row)
                train_data_en.append(row)
            index2 += 1
        train_data_en_all.append(train_data_en)

    course_track2 = {}
    for course in courses_term_1198:
        train_data_en = []
        index2 =0
        # binary_vector = [0 for x in range(len(item_list))]
        for baskets in train_data['baskets']:
            ts = train_data['timestamps'][index2]
            ls = train_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                # if ts[t]==1198 and t==len(ts)-1:
                if ts[t]==1198 and ls==1198:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(reversed_item_dict[item2]==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track2:
                    course_track2[item_dict[course]] = [row]
                else:
                    course_track2[item_dict[course]].append(row)
                train_data_en.append(row)
            index2 += 1
        train_data_en_all.append(train_data_en)
    list_of_courses = list(course_track.keys())
    for course in course_track2.keys():
        if course not in list_of_courses:
            course_track[course] = course_track2[course]
            list_of_courses.append(course)
        else: 
            course_track[course] += course_track2[course]
        # if course not in list_of_courses:
            #list_of_courses.append(course)

        
    course_track3 = {}
    for course in courses_term_1195:
        train_data_en = []
        index2 =0
        #binary_vector = [0 for x in range(len(item_list))]
        for baskets in train_data['baskets']:
            ts = train_data['timestamps'][index2]
            ls = train_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1195 and ls==1195:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(reversed_item_dict[item2]==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track3:
                    course_track3[item_dict[course]] = [row]
                else:
                    course_track3[item_dict[course]].append(row)
                train_data_en.append(row)
            index2 += 1
        train_data_en_all.append(train_data_en)
    
    for course in course_track3.keys():
            if course not in list_of_courses:
                course_track[course] = course_track3[course]
                list_of_courses.append(course)
            else: 
                course_track[course] += course_track3[course]
            # if course not in list_of_courses:
            #     list_of_courses.append(course)
    course_track4 = {}
    for course in courses_term_1191:
        train_data_en = []
        index2 =0
        #binary_vector = [0 for x in range(len(item_list))]
        for baskets in train_data['baskets']:
            ts = train_data['timestamps'][index2]
            ls = train_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1191 and ls==1191:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(reversed_item_dict[item2]==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track4:
                    course_track4[item_dict[course]] = [row]
                else:
                    course_track4[item_dict[course]].append(row)
                train_data_en.append(row)
            index2 += 1
        train_data_en_all.append(train_data_en)
    
    # new lines
    #list_of_courses = list(course_track4.keys())
    #course_track = course_track4.copy()

    for course in course_track4.keys():
            if course not in list_of_courses:
                course_track[course] = course_track4[course]
                list_of_courses.append(course)
            else: 
                course_track[course] += course_track4[course]
    
    course_track5 = {}
    for course in courses_term_1188:
        train_data_en = []
        index2 =0
        #binary_vector = [0 for x in range(len(item_list))]
        for baskets in train_data['baskets']:
            ts = train_data['timestamps'][index2]
            ls = train_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1188 and ls==1188:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(reversed_item_dict[item2]==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track5:
                    course_track5[item_dict[course]] = [row]
                else:
                    course_track5[item_dict[course]].append(row)
                train_data_en.append(row)
            index2 += 1
        train_data_en_all.append(train_data_en)
    
    for course in course_track5.keys():
            if course not in list_of_courses:
                course_track[course] = course_track5[course]
                list_of_courses.append(course)
            # else: 
            #     course_track[course] += course_track5[course]

    course_track6 = {}
    for course in courses_term_1185:
        train_data_en = []
        index2 =0
        #binary_vector = [0 for x in range(len(item_list))]
        for baskets in train_data['baskets']:
            ts = train_data['timestamps'][index2]
            ls = train_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1185 and ls==1185:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(reversed_item_dict[item2]==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track6:
                    course_track6[item_dict[course]] = [row]
                else:
                    course_track6[item_dict[course]].append(row)
                train_data_en.append(row)
            index2 += 1
        train_data_en_all.append(train_data_en)
    
    for course in course_track6.keys():
            if course not in list_of_courses:
                course_track[course] = course_track6[course]
                list_of_courses.append(course)
            else: 
                course_track[course] += course_track6[course]

    if(len(list_of_courses)!= len(item_list)):
        semester = 1185
        count_itr = 0
        while(1):
            course_track7 = {}
            if semester==1148: break
            semester, courses_term_prior = find_prior_term3(semester, train_data, term_dict_train)
            for course in courses_term_prior:
                train_data_en = []
                index2 =0
                #binary_vector = [0 for x in range(len(item_list))]
                for baskets in train_data['baskets']:
                    ts = train_data['timestamps'][index2]
                    ls = train_data['last_semester'][index2]
                    binary_vector = [0 for x in range(len(item_list))]
                    #flag = 0
                    index3 = -1
                    for t in range(len(ts)):
                        if ts[t]==semester and ls==semester:
                            index3 = t
                    if index3!= -1:
                        baskets2 = baskets[0:index3]
                        for basket in baskets2:
                            for item in basket:
                                binary_vector[item]= 1
                        t_basket = baskets[index3]
                        flag = 0
                        for item2 in t_basket:
                            if(reversed_item_dict[item2]==course):
                                flag= 1
                        row = [binary_vector, flag]
                        if item_dict[course] not in course_track7:
                            course_track7[item_dict[course]] = [row]
                        else:
                            course_track7[item_dict[course]].append(row)
                        train_data_en.append(row)
                    index2 += 1
                train_data_en_all.append(train_data_en)
            for course in course_track7.keys():
                if course not in list_of_courses:
                    course_track[course] = course_track7[course]
                    list_of_courses.append(course)
                else: 
                    course_track[course] += course_track7[course]
                # if course not in list_of_courses:
                #     list_of_courses.append(course)
            count_itr += 1
            if(len(list_of_courses)== len(item_list)): break

    # train_df = []
    # for data2 in train_data_en_all:
    #     for data3 in data2:
    #         bv, target = data3
    #         row = []
    #         #col = []
    #         for v in bv:
    #             row.append(v)
    #         row.append(target)
    #         train_df.append(row)
    # col = []
    # for i in range(len(item_list)):
    #     col.append(i)
    # col.append('target')
    # train_df = pd.DataFrame(train_df, columns= col)
    # train_df.to_csv('./train_df.csv')
    # train_df.to_json('./train_df.json', orient='records', lines=True)
    course_track = dict(sorted(course_track.items(), key=lambda item: item[0], reverse= False))

    return train_data_en_all, course_track

def prepare_test_data(test_data, item_list, item_dict, reversed_item_dict, test_target):
    #course_list = []
    # terms = []
    # for col in avg_enrollment_train.columns:
    #     terms.append(col)
    # avg_en_dict = {}
    # for x in range(len(avg_enrollment_train)):
    #     for y in terms:
    #         avg_en_dict[y] = avg_enrollment_train[y][x]

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
    #print(courses_term_1195)
    test_data_en_all = []
    course_track = {}
    for course in courses_term_1221:
        test_data_en = []
        index2 =0
        for baskets in test_data['baskets']:
            ts = test_data['timestamps'][index2]
            ls = test_data['last_semester'][index2] 
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1221 and ls==1221:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item_dict[item]]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(item2==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track:
                    course_track[item_dict[course]] = [row]
                else:
                    course_track[item_dict[course]].append(row)
                test_data_en.append(row)
            index2 += 1
        test_data_en_all.append(test_data_en)
    course_track2 = {}
    for course in courses_term_1218:
        test_data_en = []
        index2 =0
        # binary_vector = [0 for x in range(len(item_list))]
        for baskets in test_data['baskets']:
            ts = test_data['timestamps'][index2]
            ls = test_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1218 and ls==1218: 
                #if ts[t]==1218:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item_dict[item]]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(item2==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track2:
                    course_track2[item_dict[course]] = [row]
                else:
                    course_track2[item_dict[course]].append(row)
                test_data_en.append(row)
            index2 += 1
        test_data_en_all.append(test_data_en)
        
    course_track3= {}
    for course in courses_term_1215:
        test_data_en = []
        index2 =0
        #binary_vector = [0 for x in range(len(item_list))]
        for baskets in test_data['baskets']:
            ts = test_data['timestamps'][index2]
            ls = test_data['last_semester'][index2]
            binary_vector = [0 for x in range(len(item_list))]
            #flag = 0
            index3 = -1
            for t in range(len(ts)):
                if ts[t]==1215 and ls== 1215: #t==len(ts)-1
                #if ts[t]==1215:
                    index3 = t
            if index3!= -1:
                baskets2 = baskets[0:index3]
                for basket in baskets2:
                    for item in basket:
                        binary_vector[item_dict[item]]= 1
                t_basket = baskets[index3]
                flag = 0
                for item2 in t_basket:
                    if(item2==course):
                        flag= 1
                row = [binary_vector, flag]
                if item_dict[course] not in course_track3:
                    course_track3[item_dict[course]] = [row]
                else:
                    course_track3[item_dict[course]].append(row)
                test_data_en.append(row)
            index2 += 1
        test_data_en_all.append(test_data_en)
    # train_df = []
    # for data2 in train_data_en_all:
    #     for data3 in data2:
    #         train_df.append(data3)

    # train_df = pd.DataFrame(train_df, columns=['binary_vector', 'target'])
    # train_df.to_csv('./test_df.csv')
    # train_df.to_json('./test_df.json', orient='records', lines=True)
    # return train_data_en_all, train_df, course_track
    course_track = dict(sorted(course_track.items(), key=lambda item: item[0], reverse= False))
    course_track2 = dict(sorted(course_track2.items(), key=lambda item: item[0], reverse= False))
    course_track3 = dict(sorted(course_track3.items(), key=lambda item: item[0], reverse= False))
    return test_data_en_all, course_track, course_track2, course_track3

def Logistic_regression_main(train_data_en_all, course_track):
    
    list_of_models = []
    for course in course_track.keys():
        X = []
        y = []
        train_data = course_track[course]
        flag1 =0
        flag0= 0
        cnt1= 0
        for bv, target in train_data:
            row = []
            for v in bv:
                row.append(v)
            if target==1: 
                flag1=1
                cnt1+= 1
            #I will work on here
            if target==0: flag0=1
            X.append(row)
            y.append(target)
        # if flag1==1 and flag0==1 and cnt1>=5:  # at least 5 students took the course in previous semesters
        if flag1==1 and flag0==1:  # at least 5 students took the course in previous semesters
            X = np.array(X).reshape(-1,334)
            y= np.array(y)
            # scaler = preprocessing.StandardScaler().fit(X)
            # X_scaled = scaler.transform(X)
            # regr = LogisticRegression(random_state=0).fit(X_scaled, y)
            regr = LogisticRegression(random_state=0).fit(X, y)
            list_of_models.append(regr)
        else:
            list_of_models.append(regr) #prev regr with at least 10 students took the course in previous semesters
    # regr.fit(X, y)
    return list_of_models
    # pipe = make_pipeline(StandardScaler(), LogisticRegression())
    # y = train_df.iloc[:, -1]
    # pipe.fit(X, y)
    #return pipe



def evaluate_test_data(test_data_en_all, regr, course_track, course_track2, course_track3, all_courses_en, reversed_item_dict):
    # X = test_data_en_all.iloc[:, 0:-1]
    # y = test_data_en_all.iloc[:, -1]
    course_en_ac_pr = []
    act_all = []
    pred_all = []
    course_list = []
    prior_en_list = []
    for course in course_track.keys():
        X = []
        y = []
        index5 = -1
        for courses in all_courses_en['course_id']:
            if reversed_item_dict[course]==courses:
                index5 = all_courses_en[all_courses_en['course_id'] == courses].index.values[0]
        course_list.append(reversed_item_dict[course])
        if index5!= -1: n_prior_enr = all_courses_en['1218'][index5]
        else: n_prior_enr = 0
        prior_en_list.append(n_prior_enr)
        test_data = course_track[course]
        cnt1 =0
        flag1= 0
        flag0= 0
        for bv, target in test_data:
            row = []
            for v in bv:
                row.append(v)
            if target==1: 
                flag1=1
                cnt1+= 1
            if target==0: flag0=1
            X.append(row)
            y.append(target)
        X = np.array(X).reshape(-1,334)
        # scaler = preprocessing.StandardScaler().fit(X)
        # X_scaled = scaler.transform(X)
        # if course in regr:
        if cnt1>=4: # at least 4 students took the course in target semester
            regr1 = regr[course]
            prev_reg = regr1
        # else:
        #     regr1 = prev_reg
        # #en_pred = regr1.predict(X_scaled)
        # en_pred = regr1.predict(X)
        # sum1 = 0
        # sum2 = 0
        # for j in range(len(y)):
        #     if(y[j]==1):
        #         sum1+= 1
        #     if(en_pred[j]==1):
        #         sum2 += 1
        # n_actual = sum1
        # n_predicted = sum2
        # act_all.append(n_actual)
        # pred_all.append(n_predicted)
        # row = [reversed_item_dict[course], n_actual, n_predicted, n_prior_enr]
        # course_en_ac_pr.append(row)
    
    for course in course_track.keys():
        X = []
        y = []
        index5 = -1
        for courses in all_courses_en['course_id']:
            if reversed_item_dict[course]==courses:
                index5 = all_courses_en[all_courses_en['course_id'] == courses].index.values[0]
        course_list.append(reversed_item_dict[course])
        if index5!= -1: n_prior_enr = all_courses_en['1218'][index5]
        else: n_prior_enr = 0
        prior_en_list.append(n_prior_enr)
        test_data = course_track[course]
        cnt1 =0
        flag1= 0
        flag0= 0
        for bv, target in test_data:
            row = []
            for v in bv:
                row.append(v)
            if target==1: 
                flag1=1
                cnt1+= 1
            if target==0: flag0=1
            X.append(row)
            y.append(target)
        X = np.array(X).reshape(-1,334)
        # scaler = preprocessing.StandardScaler().fit(X)
        # X_scaled = scaler.transform(X)
        # if course in regr:
        if cnt1>=4:
            regr1 = regr[course]
            prev_reg = regr1
        else:
            regr1 = prev_reg
        #en_pred = regr1.predict(X_scaled)
        en_pred = regr1.predict(X)
        sum1 = 0
        sum2 = 0
        for j in range(len(y)):
            if(y[j]==1):
                sum1+= 1
            if(en_pred[j]==1):
                sum2 += 1
        n_actual = sum1
        n_predicted = sum2
        act_all.append(n_actual)
        pred_all.append(n_predicted)
        row = [reversed_item_dict[course], n_actual, n_predicted, n_prior_enr]
        course_en_ac_pr.append(row)

    for course in course_track2.keys():
        X = []
        y = []
        index5 = -1
        for courses in all_courses_en['course_id']:
            if reversed_item_dict[course]==courses:
                index5 = all_courses_en[all_courses_en['course_id'] == courses].index.values[0]
        course_list.append(reversed_item_dict[course])
        if index5!= -1: n_prior_enr = all_courses_en['1215'][index5]
        else: n_prior_enr = 0
        prior_en_list.append(n_prior_enr)
        test_data = course_track2[course]
        cnt1 =0 
        for bv, target in test_data:
            row = []
            for v in bv:
                row.append(v)
            if target==1: 
                flag1=1
                cnt1+= 1
            if target==0: flag0=1
            X.append(row)
            y.append(target)
        X = np.array(X).reshape(-1,334)
        # scaler = preprocessing.StandardScaler().fit(X)
        # X_scaled = scaler.transform(X)
        #regr1 = regr[course]
        if cnt1>=4:
            regr1 = regr[course]
            prev_reg = regr1
        else:
            regr1 = prev_reg
        #en_pred = regr1.predict(X_scaled)
        en_pred = regr1.predict(X)
        sum1 = 0
        sum2 = 0
        for j in range(len(y)):
            if(y[j]==1):
                sum1+= 1
            if(en_pred[j]==1):
                sum2 += 1
        n_actual = sum1
        n_predicted = sum2
        act_all.append(n_actual)
        pred_all.append(n_predicted)
        row = [reversed_item_dict[course], n_actual, n_predicted, n_prior_enr]
        course_en_ac_pr.append(row)

    for course in course_track3.keys():
        X = []
        y = []
        index5 = -1
        for courses in all_courses_en['course_id']:
            if reversed_item_dict[course]==courses:
                index5 = all_courses_en[all_courses_en['course_id'] == courses].index.values[0]
        course_list.append(reversed_item_dict[course])
        if index5!= -1: n_prior_enr = all_courses_en['1211'][index5]
        else: n_prior_enr = 0
        prior_en_list.append(n_prior_enr)
        test_data = course_track3[course]
        cnt1 =0
        for bv, target in test_data:
            row = []
            for v in bv:
                row.append(v)
            if target==1: 
                flag1=1
                cnt1+= 1
            if target==0: flag0=1
            X.append(row)
            y.append(target)
        X = np.array(X).reshape(-1,334)
        # scaler = preprocessing.StandardScaler().fit(X)
        # X_scaled = scaler.transform(X)
        #regr1 = regr[course]
        if cnt1>=4: # at least 4 students took the course in target semester
            regr1 = regr[course]
            prev_reg = regr1
        else:
            regr1 = prev_reg
        # en_pred = regr1.predict(X_scaled)
        en_pred = regr1.predict(X)
        sum1 = 0
        sum2 = 0
        for j in range(len(y)):
            if(y[j]==1):
                sum1+= 1
            if(en_pred[j]==1):
                sum2 += 1
        n_actual = sum1
        n_predicted = sum2
        act_all.append(n_actual)
        pred_all.append(n_predicted)
        row = [reversed_item_dict[course], n_actual, n_predicted, n_prior_enr]
        course_en_ac_pr.append(row)
   
    course_en_ac_pr = pd.DataFrame(course_en_ac_pr, columns=['courseID', 'actual_enrollment', 'predicted_enrollment', 'n_prior_enr'])
    course_en_ac_pr.to_csv('./course_en_ac_pr_Logistic_regression_v10.csv')
    course_en_ac_pr.to_json('./course_en_ac_pr_Logistic_regression_v10.json', orient='records', lines=True)

    return act_all, pred_all, course_en_ac_pr

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
    output_path1=  "./test_course_allocation_v5.txt"
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
    # course_allocation_actual_predicted.to_csv('./test_course_allocation_v2.csv')
    # return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list



if __name__ == '__main__':
#     train, test, valid = split_data('./Others/Logistic_regression_2/train_sample.csv')
   train_data = pd.read_json('./train_data_en_pred_filtered.json', orient='records', lines= True)
   #train_data = pd.read_json('./Filtered_data/train_sample_augmented.json', orient='records', lines= True)
   train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
   train_all, train_set_without_target, target, max_len = preprocess_train_data_part2(train_data) 
   term_dict_train, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses_train(train_all, reversed_item_dict)
   item_list = list(item_dict.keys())
   print("total items: ")
   print(len(item_list))
   #print(len(item_dict))
#    print(train_all)
#    print("max_len:", max_len)
   #print(target)
   #valid_data = pd.read_json('./valid_data_all.json', orient='records', lines= True)
   valid_data = pd.read_json('./Filtered_data/valid_sample_all.json', orient='records', lines= True)
   valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
   valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data) #  #
   #print("reversed_user_dict2: ", reversed_user_dict2)
   #print(valid_all)
   test_data = pd.read_json('./Filtered_data/test_sample_all.json', orient='records', lines= True)
   test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
   test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data) #, item_dict, user_dict, reversed_item_dict, reversed_user_dict #
   term_dict_test, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(test_all)
   #print(term_dict_test[1221])
   print("step 4 done")
   offered_courses = offered_course_cal('./all_data.csv')
#    train_courses = pd.read_json('./course_df.json', orient='records', lines= True)
#    avg_enrollment_train = pd.read_json('./avg_enrollment_prior_term.json', orient='records', lines= True)
   
   all_courses_en = pd.read_json('./course_df_all.json', orient='records', lines= True)
   avg_enrollment_all = pd.read_json('./avg_enrollment_prior_term_all.json', orient='records', lines= True)
   train_data_en_all, course_track = prepare_train_data(train_all, item_list, item_dict, reversed_item_dict, term_dict_train)
   # Call Logistic Regression algorithm
   regr = Logistic_regression_main(train_data_en_all, course_track)

   test_data_en_all, course_track, course_track2, course_track3 = prepare_test_data(test_all, item_list, item_dict, reversed_item_dict, test_target)
   y_test, en_pred, course_en_ac_pred = evaluate_test_data(test_data_en_all, regr, course_track, course_track2, course_track3, all_courses_en, reversed_item_dict)
   calculate_mse_for_course_allocation(y_test, en_pred)

   
