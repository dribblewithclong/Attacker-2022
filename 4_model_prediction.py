import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from joblib import load
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# Load datasets
raw_processed = pd.read_csv('attacker_raw_processed.csv')

# Load model
model = load('model.joblib')

def process_data(data):
    
    df = data.copy()

    df.columns = [i.strip() for i in df.columns]

    col_used = [
        'cat_3', 'cat_6', 'cat_8', 'cat_10', 'cat_11', 'cat_12', 'location_id', 'com_type',
        'value', 'num_date_review', 'unknown_var_7', 'unknown_var_8', 'unknown_var_9', 'unknown_var_13', 'unknown_var_17', 
        'social_friend_count', 'dob', 'time_1', 'time_2', 'date_2', 'date_3'
    ]
    df = df[col_used]

    # Process categorical data

    # Cat 3
    df['cat_3'] = df['cat_3'].apply(lambda x:'class_' + str(int(x)) if (str(x) != 'nan') & (x in range(6)) else np.nan)
    cat_3_mapping = {
        'class_0':'S','class_1':'L','class_2':'M',
        'class_3':'S1','class_4':'S','class_5':'S'
    }
    df['cat_3'] = df['cat_3'].map(cat_3_mapping)

    # Cat 6
    df['cat_6'] = df['cat_6'].apply(lambda x:'class_' + str(int(x)) if (str(x) != 'nan') & (x in range(3)) else np.nan)
    cat_6_mapping = {
        'class_0':'S','class_1':'L','class_2':'S1'
    }
    df['cat_6'] = df['cat_6'].map(cat_6_mapping)

    # Cat 8
    cat_8_mapping = {
        'BX':'S','SF':'S1','AN':'S1','QG':'S1','AS':'S1','SC':'S1',
        'TR':'S1','FJ':'S','HR':'S','EC':'S','NM':'S1','SU':'S1',
        'AY':'S1','A2':'S','BB':'S','CA':'S1','SE':'S','BE':'S1',
        'BQ':'S1','NJ':'S','NC':'S','HP':'S','FF':'S','EG':'S',
        'FI':'S','T2':'S','QP':'S','HQ':'S','FH':'S','B2':'S',
        'NE':'S1','AB':'S','QO':'S1','NG':'S','QH':'S','HN':'S1',
        'EB':'S1','QL':'S','QM':'S','NF':'S1','FE':'S','FD':'S1',
        'QD':'S1','EA':'S1','AH':'S1','QB':'S1','FC':'S1',
        'FB':'S','HO':'S','HM':'S1','BH':'S','QI':'S','QF':'S1','FG':'S1',
        'FA':'S','BT':'S1','T5':'S1','T3':'S1','DZ':'S','QE':'S',
        'QC':'S','T1':'S','AA':'S1','BA':'S','HL':'S','BP':'S',
        'T7':'S','QA':'S1','TT':'S1','YV':'S','TS':'S','TQ':'S',
        'HK':'S','AI':'M1','HJ':'M','AL':'M1','TU':'M','KC':'M',
        'TP':'M','AQ':'M','TX':'M','TV':'M','TK':'M','TY':'M',
        'HI':'M','T9':'M','HD':'M1','HC':'M1','HB':'M','HE':'M',
        'HF':'M','HG':'M','AC':'M','HW':'M','HH':'M1','BN':'M',
        'TJ':'M','TN':'M','TW':'M','S':'S','TM':'M','AO':'M',
        'TO':'M','HA':'M','AP':'M1','TH':'M','BD':'M','AR':'M',
        'TI':'M','TL':'M','HZ':'M','QZ':'M','TD':'L','TB':'L',
        'TC':'L','TE':'L','TG':'L','QW':'L','TF':'L','TZ':'L',
        'BO':'L','TA':'L','BI':'L','YN':'L'
    }
    min_cat_8 = df['cat_8'].value_counts()
    df.loc[df['cat_8'].isin(min_cat_8[min_cat_8<5].index),'cat_8'] = 'S'
    df['cat_8'] = df['cat_8'].map(cat_8_mapping)

    # Cat 10
    df['cat_10'] = df['cat_10'].replace(43,1).apply(lambda x:'class_' + str(int(x)) if (str(x) != 'nan') & (x in range(10)) else np.nan)
    cat_10_mapping = {
        'class_0':'M','class_1':'S1','class_2':'L','class_3':'M','class_4':'M',
        'class_5':'M','class_6':'S1','class_7':'S','class_8':'M1','class_9':'S1'
    }
    df['cat_10'] = df['cat_10'].map(cat_10_mapping)

    # Cat 11
    df['cat_11'] = df['cat_11'].replace(0,1).apply(lambda x:'class_' + str(int(x)) if (str(x) != 'nan') & (x in range(1,6)) else np.nan)
    cat_11_mapping = {
        'class_1':'L','class_2':'M1','class_3':'S','class_4':'S1','class_5':'M'
    }
    df['cat_11'] = df['cat_11'].map(cat_11_mapping)

    # Cat 12
    df['cat_12'] = df['cat_12'].map({'B':'class_1','C':'class_2','D':'class_3','E':'class_4','F':'class_5','G':'class_6','H':'class_7','I':'class_8'})
    cat_12_mapping = {
        'class_1':'M','class_2':'L','class_3':'L','class_4':'L',
        'class_5':'M1','class_6':'L','class_7':'S','class_8':'S1'
    }
    df['cat_12'] = df['cat_12'].map(cat_12_mapping)

    # location id
    location_id_mapping = {
        'XV':'S','CY':'S','XN':'S','HK':'S','B3':'S','HG':'S','TC':'S','CB':'S',
        'TS':'S','HD':'S','TA':'S','HX':'S','HS':'S','TK':'S','NN':'S','XD':'S',
        'NO':'S1','CK':'S1','TE':'S1','KC':'S1','HT':'S1','BT':'S1',
        'XK':'M','HN':'M','CN':'M1','DT':'M1','DK':'M1','GB':'M1','SV':'M1',
        'CH':'L','TN':'L','HC':'L','GD':'L','DN':'L'
    }
    df['location_id'] = df['location_id'].map(location_id_mapping)

    # Company type
    com_type_mapping = {
        'Vùng 1':'L','DN đầu tư NN (Vùng 1 có ĐT) {01->09, 15}':'S',
        'Vùng 2':'M','DN đầu tư NN (Vùng 2 có ĐT) {10->14, 16, 19->23, 25, 29}':'S1',
        'Vùng 3':'M','DN đầu tư NN (Vùng 3 có ĐT) {17, 18, 24, 26->28}':'S',
        'Vùng 4':'S1','DN đầu tư NN (Vùng 4 có ĐT)':'S1',
        'DN tư nhân':'S1'
    }
    df['com_type'] = df['com_type'].map(com_type_mapping)

    # Process numerical data
    
    # Transaction values
    df['value'] = df['value'].str.strip().replace('-',np.nan).apply(lambda x:float(x.replace(',','')) if str(x) != 'nan' else np.nan)
    
    # Process datetime data
    for col in ['time_1', 'time_2', 'date_2', 'date_3']:
        df[col] = pd.to_datetime(df[col])
    
    # Age
    df['dob'] = pd.to_datetime(df['dob'].fillna(9999).apply(lambda x:str(int(x))[:4]+'-'+str(int(x))[4:6]+'-'+str(int(x))[6:] if x != 9999 else np.nan))
    df['age'] = 2022 - df['dob'].dt.year

    # Delta date
    df['delta_date'] = (df['date_3'].dt.date - df['date_2'].dt.date).dt.components['days']

    # Delta time
    delta_time = (df['time_2'] - df['time_1']).dt.components
    df['delta_time'] = ((((delta_time['days']*24 + delta_time['hours'])*60 + delta_time['minutes'])/60)/24).round(3)

    # Remove redundant columns
    df.drop(['dob', 'time_1', 'time_2', 'date_2', 'date_3'], axis=1, inplace=True)

    return df

def fill_missing_numerical(raw_processed, feature, data):

        numerical_categorical_group = raw_processed.groupby(list(raw_processed.select_dtypes('object').columns))[feature].median().reset_index()
        numerical_categorical_group[feature] = numerical_categorical_group[feature].fillna(numerical_categorical_group[feature].median())

        missing_feature_data = data[data[feature].isnull()][list(data.select_dtypes('object').columns) + [feature]]
        for col in missing_feature_data.columns:
            if col != feature:
                missing_feature_data[col] = missing_feature_data[col].fillna(raw_processed[col].mode().values[0])
                
        categorical_group_available = missing_feature_data.select_dtypes('object').drop_duplicates()
        for i in categorical_group_available.dropna().iterrows():
            cat_3 = i[1]['cat_3']
            cat_6 = i[1]['cat_6']
            location_id = i[1]['location_id']
            cat_8 = i[1]['cat_8']
            cat_10 = i[1]['cat_10']
            cat_11 = i[1]['cat_11']
            com_type = i[1]['com_type']
            cat_12 = i[1]['cat_12']
            try:
                missing_feature_data.loc[
                    (missing_feature_data['cat_3'] == cat_3) &
                    (missing_feature_data['cat_6'] == cat_6) &
                    (missing_feature_data['location_id'] == location_id) &
                    (missing_feature_data['cat_8'] == cat_8) &
                    (missing_feature_data['cat_10'] == cat_10) &
                    (missing_feature_data['cat_11'] == cat_11) &
                    (missing_feature_data['com_type'] == com_type) &
                    (missing_feature_data['cat_12'] == cat_12),
                    feature
                ] =  numerical_categorical_group.loc[
                    (numerical_categorical_group['cat_3'] == cat_3) &
                    (numerical_categorical_group['cat_6'] == cat_6) &
                    (numerical_categorical_group['location_id'] == location_id) &
                    (numerical_categorical_group['cat_8'] == cat_8) &
                    (numerical_categorical_group['cat_10'] == cat_10) &
                    (numerical_categorical_group['cat_11'] == cat_11) &
                    (numerical_categorical_group['com_type'] == com_type) &
                    (numerical_categorical_group['cat_12'] == cat_12),
                    feature
                ].values[0]
            except:
                pass

        missing_feature_data.dropna(inplace=True)
        
        data.loc[missing_feature_data.index,feature] = missing_feature_data[feature]
        
        data[feature] = data[feature].fillna(raw_processed[feature].median())

        return data

def fill_missing_categorical(raw_processed, feature, data):

    train = raw_processed[list(raw_processed.select_dtypes('float').columns) + [feature]].dropna()

    missing_feature_data = data.loc[data[feature].isnull()][list(data.select_dtypes('float').columns) + [feature]]
    
    if len(missing_feature_data) == 0:
        return data
    
    tree = DecisionTreeClassifier(random_state=44,criterion='log_loss')
    tree.fit(train.drop(feature,axis=1),train[feature])
    
    data.loc[missing_feature_data.index,feature] = tree.predict(missing_feature_data.drop(feature,axis=1))
    
    return data

def encode_data(raw_processed, data):

    raw_category = raw_processed.select_dtypes('object')
    data_category = data.select_dtypes('object')

    OHE = OneHotEncoder(drop='first',sparse=False,handle_unknown='ignore')
    OHE.fit(raw_category)

    raw_OHE = pd.DataFrame(OHE.transform(raw_category),columns=OHE.get_feature_names_out(),index=raw_category.index)
    data_OHE = pd.DataFrame(OHE.transform(data_category),columns=OHE.get_feature_names_out(),index=data_category.index)

    pca = PCA(n_components=8, random_state=44)
    pca.fit(raw_OHE)

    data_PCA = pd.DataFrame(pca.transform(data_OHE),index=data_OHE.index,columns=[f'pca_{i}' for i in range(1,9)])

    data = pd.merge(data.select_dtypes(exclude='object'), data_PCA, left_index=True, right_index=True)\
    
    # Sort data columns
    data = data[[
        'value', 'num_date_review', 'unknown_var_7', 'unknown_var_8', 'unknown_var_9', 'unknown_var_13', 'unknown_var_17',
        'social_friend_count', 'delta_time', 'delta_date', 'age', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8'
    ]]

    return data

def data_preparation(data, raw_processed):

    print('Begin processing data')
    data_processed = process_data(data)
    raw_processed = raw_processed[list(data_processed.columns)]
    print('Completed processing data\n')

    print('Begin filling missing data')
    for col in data_processed.select_dtypes('float').columns:
        data_processed = fill_missing_numerical(raw_processed, col, data_processed)
        print(f'Filled {col}')

    for col in data_processed.select_dtypes('object').columns:
        data_processed = fill_missing_categorical(raw_processed, col, data_processed)
        print(f'Filled {col}')
    print('Completed filling missing data\n')

    print('Begin encoding data')
    data_processed = encode_data(raw_processed, data_processed)
    print('Completed encoding data\n')

    print('**********************')
    print('Data is prepared to be predicted')
    print('**********************')

    return data_processed

def model_predictions(data):
    predicted = model.predict(data)

    return predicted

def predict_from_user(
    cat_3, cat_6, cat_8, cat_10, cat_11, cat_12, location_id, com_type,
    value, num_date_review, unknown_var_7, unknown_var_8, unknown_var_9,
    unknown_var_13, unknown_var_17, social_friend_count, dob, time_1,
    time_2, date_2, date_3
):
    
    new_data = pd.DataFrame([[
        cat_3, cat_6, cat_8, cat_10, cat_11, cat_12, location_id, com_type,
        value, num_date_review, unknown_var_7, unknown_var_8, unknown_var_9,
        unknown_var_13, unknown_var_17, social_friend_count, dob, time_1,
        time_2, date_2, date_3
    ]], columns=[
        'cat_3', 'cat_6', 'cat_8', 'cat_10', 'cat_11', 'cat_12', 'location_id', 'com_type',
        'value', 'num_date_review', 'unknown_var_7', 'unknown_var_8', 'unknown_var_9', 'unknown_var_13', 'unknown_var_17', 
        'social_friend_count', 'dob', 'time_1', 'time_2', 'date_2', 'date_3'
    ])

    prepared_data = data_preparation(new_data, raw_processed)
    
    predicted_data = model_predictions(prepared_data)

    return predicted_data

prediction = predict_from_user(
    cat_3=1,
    cat_6=1,
    cat_8='HG',
    cat_10=8,
    cat_11=0,
    cat_12='C',
    location_id='HC',
    com_type='Vùng 4',
    value='7,014,920',
    num_date_review=245.0,
    unknown_var_7=28.0,
    unknown_var_8=17.677670,
    unknown_var_9=15.500000,
    unknown_var_13=0.91,
    unknown_var_17=3993.0,
    social_friend_count=np.nan,
    dob=19751110.0,
    time_1='2018-12-19T01:59:10.7Z',
    time_2='2018-12-19T08:32:24Z',
    date_2='12/19/2018',
    date_3='1/1/2015'
)

print(prediction)

