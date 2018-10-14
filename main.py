
## 匯入套件
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


## 匯入訓練資料
buy = pd.read_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/train_buy_info.csv"))
cust = pd.read_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/train_cust_info.csv"))
tpy = pd.read_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/train_tpy_info.csv"))

## 觀察哪一類保險商品購買數最多
# 保險商品購買類別以 a / d / e 為最多
tmp = buy["BUY_TYPE"].value_counts().reset_index().rename(columns={"index":"BUY_TYPE", "BUY_TYPE":"mean"})
tmp["mean"] = tmp["mean"]/tmp["mean"].sum()
tmp["mean_upper"] = tmp["mean"].apply(lambda x:x*1.3)
tmp["mean_lower"] = tmp["mean"].apply(lambda x:x*0.7)
check = tmp.drop("mean", axis=1)

# 	BUY_TYPE	mean	mean_upper	mean_lower
# 0		e	0.276732	0.359751	0.193712
# 1		a	0.275119	0.357655	0.192584
# 2		d	0.274928	0.357406	0.192449
# 3		c	0.110547	0.143711	0.077383
# 4		f	0.029551	0.038416	0.020685
# 5		b	0.029507	0.038359	0.020655
# 6		g	0.003616	0.004701	0.002532



## 資料集非常多類別變數，需要對類別變數進行挑選，再使用 one-hot encoding
# 本資料集許多類別變數有兩個特點：

# 資料數目集中在某一個類別
# 單一類別集中購買 a/d/e 三種保險商品類別，因為 a/d/e 購買商品人數本來就多
# 以下以 cust 資料集下的 BEHAVIOR_3 變數 a 類別為例，發現資料集中在 a/d/e 三種保險購買商品，且分配和 BUY_TYPE 十分類似，因此本組認為該變數對於預測幫助不大。
mean = dataset.loc[dataset["BEHAVIOR_3"]=="a", ["BEHAVIOR_3","BUY_TYPE"]].groupby(by="BUY_TYPE").agg("count").apply(lambda x: x/x.sum())
mean.rename(columns={"BEHAVIOR_3":"BEHAVIOR_3_a_mean"}, inplace=True)

count = dataset.loc[dataset["BEHAVIOR_3"]=="a", ["BEHAVIOR_3","BUY_TYPE"]].groupby(by="BUY_TYPE").agg("count")
count.rename(columns={"BEHAVIOR_3":"BEHAVIOR_3_a_count"}, inplace=True)

pd.merge(mean, count, on="BUY_TYPE")

# 			BEHAVIOR_3_a_mean	BEHAVIOR_3_a_count
# BUY_TYPE		
# a			0.305699	12862
# b			0.000333	14
# c			0.106170	4467
# d			0.336788	14170
# e			0.222655	9368
# f			0.026026	1095
# g			0.002329	98


## 類別變數挑選標準函數：check_dummy(var)
# 本組類別變數挑選有兩種標準：

# 類別變數下，單一類別樣本數量大於 10000 或小於 160000
# 類別變數下，單一類別於 BUY_TYPE 的分佈，大於 BUY_TYPE 分佈 30% 或小於 30%
dataset = pd.merge(buy, tpy, on="CUST_ID", how="left")
dataset = pd.merge(dataset, cust, on="CUST_ID", how="left")

def check_dummy(var):
    check_list = []
    tmp = dataset[var].value_counts().to_frame().reset_index()
    for i in range(0, len(tmp.index)):
        if (tmp.iloc[i,1]<=160000) and (tmp.iloc[i,1]>=10000):
            check_list.append(tmp.iloc[i,0])
    for ctgy in check_list:
        tmp = dataset.loc[(dataset[var]==ctgy), [var,'BUY_TYPE']].groupby(['BUY_TYPE']).agg(['count']).apply(lambda x: x/x.sum())
        tmp.columns = tmp.columns.droplevel()
        merged = pd.merge(check, tmp, on="BUY_TYPE", how="inner")
        eles = merged["BUY_TYPE"].tolist()
        merged["check"] = 0
        merged.loc[(merged["count"]<=merged["mean_lower"])|(merged["count"]>=merged["mean_upper"]),"check"] = 1
        if merged["check"].sum() > 0:
            return [var, ctgy]
        else:
            pass
        
output_list = []
for var in dataset.columns.drop(["CUST_ID", "BUY_TYPE", "HEIGHT", "WEIGHT", "BUDGET"]):
    try:
        output_list.append(check_dummy(var))
    except Exception as e:
        pass


## 創建最終資料集函數：create_dataset(dataset1, dataset2, dataset3)
def create_dataset(dataset1, dataset2, dataset3):
    
    # 資料集對接
    dataset = pd.merge(dataset1, dataset2, on="CUST_ID", how="left")
    dataset = pd.merge(dataset, dataset3, on="CUST_ID", how="left")
    
    # 用中位數取代數值變數的遺漏值
    # 對數值變數進行標準化
    for i in ["HEIGHT", "WEIGHT", "BUDGET"]:
        dataset[i] = dataset[i].fillna(pd.to_numeric(dataset[i]).median())
        dataset[i] = preprocessing.scale(dataset[i])
    
    # 將挑選出來的類別變數串入最終資料集
    tmp = dataset["AGE"].to_frame()
    for i in output_list:
        try:
            var_name = str(i[0])+"_"+str(i[1])
            var_name_df = pd.get_dummies(dataset[i[0]])[i[1]].to_frame().rename(columns={i[1]:var_name})
            tmp = tmp.join(var_name_df)
        except TypeError as e:
            pass
    df_X = tmp.join(dataset[["HEIGHT", "WEIGHT", "BUDGET"]]).drop("AGE", axis=1)
    
    try:
        df_y = dataset["BUY_TYPE"]
        return [df_X, df_y]
    except:
        return df_X
    

## 利用最終資料集創造訓練集和測試集
df_X, df_y = create_dataset(buy, cust, tpy)
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3)
print("最後使用的變數")
print(df_X.columns.tolist())

# 最後使用的變數
# ['AGE_a', 'SEX_b', 'OCCUPATION_t28', 'CHILD_NUM_2', 'BUY_MONTH_10', 'MARRIAGE_b', 'BUY_TPY1_NUM_CLASS_G', 
# 'BUY_TPY2_NUM_CLASS_G', 'BUY_TPY3_NUM_CLASS_F', 'BUY_TPY4_NUM_CLASS_F', 'BUY_TPY6_NUM_CLASS_G', 'BUY_TPY7_NUM_CLASS_G', 
# 'BEHAVIOR_3_c', 'STATUS1_a', 'STATUS2_a', 'STATUS3_a', 'STATUS4_b', 'IS_NEWSLETTER_b', 'CHARGE_WAY_b', 'IS_EMAIL_b', 'IS_PHONE_a', 'INTEREST2_a', 
# 'INTEREST3_b', 'INTEREST4_b', 'INTEREST5_b', 'INTEREST6_a', 'INTEREST7_a', 'INTEREST10_b', 'IS_SPECIALMEMBER_b', 'HEIGHT', 'WEIGHT', 'BUDGET']

## 利用支援向量機進行訓練
clf = SVC(kernel="linear")
clf.fit(X_train, y_train.values.ravel())

# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)


## 進行 cross validation
# 將資料切分為 10 等分進行交互驗證，平均準確率落在 88% 左右
scores = cross_val_score(clf,df_X,df_y.values.ravel(),cv=10,scoring='accuracy')
print(scores)
print(scores.mean())

# [0.88142017 0.88048972 0.8820915  0.88122004 0.88518519 0.88453159
#  0.88440087 0.87986405 0.88069197 0.88169419]
# 0.8821589288449767


## 進行預測
buy_test = pd.read_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/test_buy_x_info.csv"))
cust_test = pd.read_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/test_cust_x_info.csv"))
tpy_test = pd.read_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/test_tpy_x_info.csv"))
dataset_test_processed = create_dataset(buy_test, cust_test, tpy_test)
prediction = clf.predict(dataset_test_processed)

## 匯出資料
Submmit_Sample_testing_Set = pd.read_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/Submmit_Sample_testing_Set.csv"))
Submmit_Sample_testing_Set.drop("BUY_TYPE", axis=1, inplace=True)
Submmit_Sample_testing_Set["BUY_TYPE"] = prediction
Submmit_Sample_testing_Set.to_csv(os.path.expanduser("~/Desktop/初賽資料(20180828)/cobers.csv"), index=False)       
Submmit_Sample_testing_Set.head()

