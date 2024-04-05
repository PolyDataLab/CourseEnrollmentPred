import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import random

#preprocess train data, create item dict and user dict 
#def split_data(input_file):
def preprocess_train_data_part1(input_data):
    data = input_data
    data["userID"] = data["userID"].fillna(-1).astype('int32')
    data=data[data["userID"]!=-1].reset_index(drop=True)

    itemIDs = {}
    index=0
    for baskets in data['baskets']:
        new_baskets = []
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)
                new_basket.append(itemIDs[item])
            new_baskets.append(new_basket)
        data['baskets'][index] = new_baskets
        index +=1                

    
    reversed_item_dict = dict(zip(itemIDs.values(), itemIDs.keys()))
    # print("reversed_item_dict: ", reversed_item_dict)
    user_dict = {}
    index = 0
    len1 =0
    reversed_user_dict = {}
    for user in data["userID"]:
        user_dict[user] = len1
        data['userID'][index] = user_dict[user]
        index += 1
        len1+=1
        reversed_user_dict[user_dict[user]]= user
    return data, itemIDs, user_dict, reversed_item_dict, reversed_user_dict

#split training set into train set without target semesters and only target semesters.    
def preprocess_train_data_part2(input_data):

    # baskets['num_baskets'] = baskets.baskets.apply(len)
    data = input_data
    users = data.userID.values
    max_len = 0
    train_valid = []
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        max_len = max(max_len, data.iloc[index]['num_baskets'])
        row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps'] ]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        train_valid.append(row)
        
    train_set_all = pd.DataFrame(train_valid, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])

    train_valid2 = []
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        #if baskets.iloc[index]['num_baskets']>=2:
        row = [user, b, data.iloc[index]['num_baskets']-1, data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        train_valid2.append(row)
        
    train_set_without_target = pd.DataFrame(train_valid2, columns=['userID', 'baskets', 'num_baskets',  'last_semester', 'timestamps'])

    target_set = []
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][-1]
        #if baskets.iloc[index]['num_baskets']>=2:
        row = [user, b, data.iloc[index]['last_semester']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        target_set.append(row)
    
    target_set = pd.DataFrame(target_set, columns=['userID', 'baskets',  'last_semester'])
   
    train_set_all.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/train_sample_all.json', orient='records', lines=True)
    train_set_without_target.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/train_set_without_target.json', orient='records', lines=True)
    target_set.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/target_set.json', orient='records', lines=True)
    return train_set_all, train_set_without_target, target_set, max_len
    #return train_set_all, train_set_without_target, target_set, max_len

def preprocess_valid_data_part1(input_data, reversed_user_dict, item_dict): #
    data = input_data
    
    len1 = len(reversed_user_dict)
    user_dict2 = {}
    index = 0
    reversed_user_dict2 = {}
    for user in data["userID"]:
        user_dict2[user] = len1
        data['userID'][index] = user_dict2[user]
        index += 1
        len1+=1
        reversed_user_dict2[user_dict2[user]]= user
    # reversed_user_dict2 = dict(zip(user_dict2.values(), user_dict2.keys()))


    #reversed_user_dict2 = dict(zip(user_dict2.values(), user_dict2.keys()))
    #print("reversed_user_dict: ", reversed_user_dict)
    #data["userID"] = data["userID"].map(users)

    return data, user_dict2, reversed_user_dict2

def preprocess_valid_data_part2(input_data):
    data = input_data
    users = data.userID.values
    valid_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_all.append(row)
        index += 1
    valid_set_all = pd.DataFrame(valid_all, columns=['userID', 'baskets', 'num_baskets',  'last_semester', 'timestamps'])
    valid_2 = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']-1, data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
            valid_2.append(row)
        index += 1
   
    valid_set_without_target = pd.DataFrame(valid_2, columns=['userID', 'baskets', 'num_baskets',  'last_semester', 'timestamps'])
    validation_target_set = []
    index = 0
    for user in data.userID.values:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][-1]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['last_semester']]
            validation_target_set.append(row)
        index += 1
    validation_target_set = pd.DataFrame(validation_target_set, columns=['userID', 'baskets',  'last_semester'])
    
    valid_set_all.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/valid_sample_all.json', orient='records', lines=True)
    valid_set_without_target.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/valid_sample_without_target.json', orient='records', lines=True)
    validation_target_set.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/validation_target_set.json', orient='records', lines=True)
    return valid_set_all, valid_set_without_target, validation_target_set
    #return valid_set_all, valid_set_without_target, validation_target_set

def preprocess_test_data_part1(input_data, reversed_user_dict, item_dict, reversed_user_dict2): #  
  
    data = input_data
    
    #reversed_user_dict = dict(zip(user_dict.values(), user_dict.keys()))
    len1 = len(reversed_user_dict) + len(reversed_user_dict2)
    user_dict3 = {}
    index = 0
    reversed_user_dict3 = {}
    for user in data["userID"]:
        user_dict3[user] = len1
        data['userID'][index] = user_dict3[user]
        index += 1
        len1+=1
        reversed_user_dict3[user_dict3[user]]= user

    #reversed_user_dict3 = dict(zip(user_dict3.values(), user_dict3.keys()))
    # baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  #
    # baskets.columns = ['userID', 'baskets']

    #baskets['num_baskets'] = baskets.baskets.apply(len)

    return data, user_dict3, reversed_user_dict3

def preprocess_test_data_part2(input_data):
    data = input_data
    users = data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
            test_all.append(row)
        index +=1
    test_set_all = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])

    test_2 = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']-1, data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
            test_2.append(row)
        index +=1
   
    test_set_without_target = pd.DataFrame(test_2, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_target_set = []
    index = 0
    for user in data.userID.values:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][-1]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['last_semester']]
            test_target_set.append(row)
        index +=1
    test_target_set = pd.DataFrame(test_target_set, columns=['userID', 'baskets', 'last_semester'])
   
    test_set_all.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/test_sample_all.json', orient='records', lines=True)
    test_set_without_target.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/test_sample_without_target.json', orient='records', lines=True)
    test_target_set.to_json('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/test_target_set.json', orient='records', lines=True)
    return test_set_all, test_set_without_target, test_target_set
    #return test_set_all, test_set_without_target, test_target_set

if __name__ == '__main__':
   #train, test, valid = split_data('/Users/mkhan149/Downloads/Experiments/Others/LSTM_R/train_sample.csv')
   #train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/train_sample_augmented.json', orient='records', lines= True)
   train_data = pd.read_json('/Users/mkhan149/Downloads/Experiments/train_data_en_pred_filtered.json', orient='records', lines= True)
   train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
   print(len(item_dict))
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
   test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
   test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data) #, item_dict, user_dict, reversed_item_dict, reversed_user_dict #, 
