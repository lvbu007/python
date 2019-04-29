# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:24:07 2019

@author: darendai
"""


"""
ssh zhuqingwen@10.19.167.17
password:zhuqingwen@123
ssh zhuqingwen@10.19.166.25
password:zqw@2017_1030&

cd /data/data3/hive_data
cd命令：更换目录
cd ~ : 切换到用户目录
cd .. ：返回到上一层目录
cd ../.. ：返回到上二层目录
ls 命令：列出文件
ls -la 列出当前目录下的所有文件和文件夹
ls a* 列出当前目录下所有以a字母开头的文件
ls -l *.txt 列出当前目录下所有后缀名为txt的文件

vim  fqz_score_dataset_02val.csv
vim  fqz_score_dataset_03train.csv
vim v3train_corr_matrix
vim corr_matrix_v5_train.csv
ctrl+c  :q! 返回命令行

yarn application -list
yarn application -kill <applicationId>
yarn application -list -appStates KILLED
yarn logs -applicationId <applicationId>
yarn queue -status szoffline

http://namenodestandby.lakala.com:8088/cluster

"""

#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:55:11 2018

@author: steven
"""

"""
第一轮训练     lkl_card_score.overdue_result_all_new_woe_instant_v3
第一轮0304验证 lkl_card_score.overdue_result_all_new_woe_instant_v3_01val
第一轮0506验证 lkl_card_score.fqz_score_dataset_01val_170506

第二轮训练     lkl_card_score.overdue_result_all_new_woe_instant_v3_02train
第二轮0304验证 lkl_card_score.fqz_score_dataset_02val
第二轮0506验证 lkl_card_score.fqz_score_dataset_02val_170506

fqz_score_dataset_05val_level95s.csv

fqz_score_dataset_05train_kill_level95.csv
fqz_score_dataset_05val_label2_noyczs.csv
fqz_score_dataset_05val_msjr_noyczs.csv

fqz_score_dataset_06val_level95s.csv

fqz_score_dataset_06train_kill_level95.csv
fqz_score_dataset_06val_msjr_noyczs.csv

fqz_score_dataset_03val_label2.csv
fqz_score_dataset_04val_label2s.csv
"""
import pandas as pd
import numpy as np
import sklearn
import sys
import sas7bdat
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
    
path = r'E:/工作资料/dataset/scoremodel/v22'
df = sas7bdat.SAS7BDAT(path,encoding='gb2312').to_data_frame()    
filename_v22train = r'/data/data3/hive_data/overdue_result_all_new_woe_instant_v3.csv'
#filename_v2train = r'/data/data3/hive_data/overdue_result_all_new_woe_instant_v3_02train.csv'
#filename_v3train = r'/data/data3/hive_data/fqz_score_dataset_03train.csv'
#filename_v3test = r'/data/data3/hive_data/fqz_score_dataset_03val.csv'
#filename_v3train_noycz = r'/data/data3/hive_data/fqz_score_dataset_03train_noycz.csv'
#filename_v3test_noycz = r'/data/data3/hive_data/fqz_score_dataset_03val_noycz.csv'
#filename_v5train = r'/data/data3/hive_data/fqz_score_dataset_05train_kill_level95.csv'
#filename_v5test = r'/data/data3/hive_data/fqz_score_dataset_05val_level95s.csv'

#data = pd.read_csv(filename,index_col='order_src',delimiter='\001')
data_v3train = pd.read_csv(filename_v3train,delimiter='\001')
#data_v3test = pd.read_csv(filename_v3test,delimiter='\001')
#data_v3train_noycz = pd.read_csv(filename_v3train_noycz,delimiter='\001')
#data_v3test_noycz = pd.read_csv(filename_v3test_noycz,delimiter='\001')
#data_v5train = pd.read_csv(filename_v5train,delimiter='\001')
#data_v5test = pd.read_csv(filename_v5test,delimiter='\001')

#data.dtypes #查看各行的数据格式
#type(data)
#data1.columns #查看列名
#data.index #查看索引
#data1.head() #查看前几行的数据,默认前5行
#data.tail() #查看后几行的数据,默认后5行
#data.values #查看数据值
#data1.describe #描述性统计
#data1.T #转置
#data1.sort(columns ='')#按列名进行排序
#data1.sort_index(by=['',''])#多列排序,使用时报该函数已过时,请用sort_values
#data1.sort_values(by=['',''])同上
#使用DataFrame选择数据(类似SQL中的LIMIT):
#data1['label'] #显示列名下的数据
#data1['f_pass_cnt']
#data1[1:3] #获取1-2行的数据,该操作叫切片操作,获取行数据
#data1[0:3] #获取0,1,2行数据
#data1.loc[:,['label','f_pass_cnt']] #获取选择区域内的数据,逗号前是行范围,逗号后是列范围,注loc通过标签选择数据,iloc通过位置选择数据
#使用pandas合并数据集(类似SQL中的JOIN):
#merge(mxj_obj2, mxj_obj1 ,on='用户标识',how='inner')# mxj_obj1和mxj_obj2将用户标识当成重叠列的键合并两个数据集,inner表示取两个数据集的交集.
#drop函数的使用：数据类型转换
#df['Name'] = df['Name'].astype(np.datetime64)
#使用DataFrame模糊筛选数据(类似SQL中的LIKE):
#df_obj[df_obj['套餐'].str.contains(r'.*?语音CDMA.*')] #使用正则表达式进行模糊匹配,*匹配0或无限次,?匹配0或1次
#data[data['order_src'].str.contains(r'XNW')]
data1=data_v5train[data_v5train.label !=2] #类似SQL where语句
#data1.drop(['order_src','apply_time','ljmx'],axis=1, inplace=True)
data1.drop(['ljmx'],axis=1, inplace=True)
data1.replace('\N',0,inplace=True)
#data1.fillna(0) #用0填充无效值NaN
#dd=data1.iloc[:,2]
#dd=data1[u'ljmx']
#DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
#data1_train=data1.sample(n=None, frac=0.7, replace=False, weights=None, random_state=None, axis=None
y=data1[u'label']
data1.drop(['label'],axis=1,inplace=True)
data_v3train1=data_v3train[:]
data_v3train1.drop(["f_order_cnt",
 "f_id_cnt",
 "f_black_cnt",
 "f_q_refuse_cnt",
 "f_pass_cnt",
 "f_2_self_tmp_overdue0",
 "f_2_self_tmp_overdue3",
 "f_2_self_tmp_overdue30",
 "f_2_self_tmp_overdue0_ls",
 "f_2_self_tmp_overdue3_ls",
 "f_2_self_tmp_overdue30_ls",
 "cnt_q61_self",
 "cnt_q62_self",
 "cnt_q65_self",
 "cnt_q66_self",
 "cnt_q67_self",
 "cnt_q68_self",
 "cnt_q72_self",
 "cnt_q79_self",
 "cnt_x_self",
 "cnt_x62_self",
 "cnt_x69_self",
 "cnt_x67_self",
 "cnt_x81_self",
 "cnt_x84_self",
 "avg_af_score_self",
 "cnt_dt35_self",
 "cnt_dt36_self",
 "cnt_dt37_self",
 "cnt_dt38_self",
 "cnt_dt63_self",
 "cnt_dt64_self",
 "cnt_dt74_self",
 "cnt_dt75_self",
 "cnt_dt76_self",
 "f_1_id_cnt",
 "f_1_2_id_cnt",
 "cnt_appmob_one",
 "cnt_appmob_two",
 "edge_woe_sum",
 "edge_woe_max",
 "edge_woe_min",
 "edge_woe_sum_w",
 "edge_woe_max_w",
 "cnt_lbs_one"],axis=1, inplace=True)
data_v3train1.drop(['ljmx'],axis=1, inplace=True)
data_v3train1.replace('\N',0,inplace=True)
y_v4=data_v3train1[u'label']
data_v3train1.drop(['label'],axis=1,inplace=True)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data1, y, test_size=0.3,random_state = 10)
x_train_v4, x_test_v4, y_train_v4, y_test_v4 = train_test_split(data_v3train1, y_v4, test_size=0.3,random_state = 10)
x = x_train.iloc[:,2:1216].as_matrix() #截取第1-199列，第一列为0，不包含索引列
x_v4 = x_train_v4.iloc[:,2:155].as_matrix()
#x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.4, random_state=0)
#y = data1.iloc[:,2].as_matrix()
#data1.isnull().sum() #sum是求的每一列的和

zqw=data1.iloc[:,2:1216]
corr_matrix_v5=data1.iloc[:,2:1216].corr()
corr_matrix_v5_train=x_train.iloc[:,2:1216].corr()
corr_matrix_v5_rlr=datax[datax.columns[rlr.get_support()]].corr()
corr_matrix_v4=data_v3train1.iloc[:,2:155].corr()
corr_matrix[u'cnt_x67_one']
# cnt_x67_one 一度_X67标拒绝数量
corr_matrix.to_csv('/data/data3/hive_data/v3train_corr_matrix.csv')
corr_matrix_v4.to_csv('/data/data3/hive_data/v4train_corr_matrix.csv')
corr_matrix_v5.to_csv('/data/data3/hive_data/corr_matrix_v5.csv')
corr_matrix_v5_train.to_csv('/data/data3/hive_data/corr_matrix_v5_train.csv')
corr_matrix_v5_rlr.to_csv('/data/data3/hive_data/corr_matrix_v5_rlr.csv')
#x_train1=x_train.iloc[:,2:201].as_matrix()
#y_train1=y_train.iloc[:].as_matrix()

"""
import csv
withopen('eggs.csv','wb')
 as csvfile:
    #spamwriter
 = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter=csv.writer(csvfile,
 dialect='excel')

f = open("text.txt",'wb')
f.write(result)
f.close()
"""

############################################################逻辑回归#############################################################
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
rlr = RLR()
rlr.fit(x, y_train)
rlr.get_support()
datax=x_train.iloc[:,2:1216]
print(u'RLR variable screen over ')
print(u'effective variable is %s' % ','.join(datax.columns[rlr.get_support()]))
x_rlr = datax[datax.columns[rlr.get_support()]].as_matrix()

"""
RandomizedLogisticRegression(C=1, fit_intercept=True, memory=None, n_jobs=1,
               n_resampling=200, normalize=True, pre_dispatch='3*n_jobs',
               random_state=None, sample_fraction=0.75, scaling=0.5,
               selection_threshold=0.25, tol=0.001, verbose=False)
			   
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
"""

lr = LR()
lr.fit(x, y_train)
lr_rlr = LR()
lr_rlr.fit(x_rlr,y_train)
lr_v4 = LR()
lr_v4.fit(x_v4, y_train_v4)
print(u'model train finish')
print(u'model avg right rate is %s' % lr.score(x, y_train)) #给出模型的平均正确率 0.884423364193 
print(u'model avg right rate is %s' % lr_rlr.score(x_rlr, y_train)) #给出模型的平均正确率 0.886172727577 0.88612709201
print(u'model avg right rate is %s' % lr_v4.score(x_v4, y_train_v4)) #给出模型的平均正确率 0.820956460627
print('Coefficient: n', lr.intercept_,lr.coef_) #导出模型回归系数
print('Coefficient: n', lr_v4.intercept_,lr_v4.coef_)
print('Coefficient: n', lr_rlr.intercept_,lr_rlr.coef_)

"""
LogisticRegression类中的方法有如下几种:
score=lr.decision_function(x) #Predict confidence scores for samples,计算样本点到分割超平面的函数距离。
lr.get_params() #Get parameters for this estimator
lr.get_params(deep=True)
predict(X) #Predict class labels for samples in X.    用来预测测试样本的标记，也就是分类。X是测试样本集 
predict_log_proba(X) #Log of probability estimates. 
predict_proba(X) #Probability estimates. 
score(X, y[, sample_weight]) #Returns the mean accuracy on the given test data and labels. 
sparsify() #Convert coefficient matrix to sparse format
transform(X[, threshold]) #Reduce X to its most important features

"""

#保存模型
from sklearn.externals import joblib
joblib.dump(lr,r"/data/data3/hive_data/lr_v3.model")
joblib.dump(lr_rlr,r"/data/data3/hive_data/lr_v3_rlr.model")
joblib.dump(lr_v4,r"/data/data3/hive_data/lr_v4.model")

#加载模型
LR_v3=joblib.load(r"/data/data3/hive_data/lr_v3.model")
LR_v3_RLR=joblib.load(r"/data/data3/hive_data/lr_v3_rlr.model")
LR_v4=joblib.load(r"/data/data3/hive_data/lr_v4.model")
#应用模型进行预测
data1_pre=x_train[:]
predict_cols = data1_pre.columns[2:]
data2_pre=x_train_v4[:]
predict2_cols = data2_pre.columns[2:]
data3_pre=x_train.loc[:,['order_src',
'apply_time',
'label',
'f_id_cnt',
'f_black_cnt',
'f_pass_cnt',
'f_2_self_tmp_overdue0_ls',
'f_2_self_tmp_overdue3_ls',
'f_2_self_tmp_overdue30_ls',
'f_1_id_cnt',
'f_1_black_cnt',
'f_2_2_tmp_overdue3',
'f_2_2_tmp_overdue30',
'edge_woe_sum',
'edge_woe_max',
'edge_woe_min',
'edge_woe_sum_w',
'depth',
'cnt_x_self',
'cnt_x69_self',
'cnt_x67_self',
'avg_af_score_self',
'avg_af_score_one',
'cnt_x_one_w',
'cnt_x62_one_w',
'cnt_x67_one_w',
'cnt_dt38_one_w',
'cnt_pass_one_w',
'cnt_cur_overdue0_one_w',
'cnt_bankcard_one',
'cnt_logmob_one',
'cnt_recom_one',
'cnt_addbook_one',
'cnt_calllog_one']]
predict3_cols = data3_pre.columns[:]
# 进行预测，并将预测评分存入pre的 predict 列中
result_v3_train=LR_v3.predict_proba(data1_pre[predict_cols])
data1_pre['predict_1']=result_v3_train[:,1]
data1_pre['label']=y_train[:]

result_v4_train=LR_v4.predict_proba(data2_pre[predict2_cols])
data2_pre['predict_1']=result_v4_train[:,1]
data2_pre['label']=y_train_v4[:]

result_v3_rlr_train=LR_v3_RLR.predict_proba(data3_pre[predict3_cols])
data3_pre['predict_1']=result_v3_rlr_train[:,1]
data3_pre['label']=y_train[:]
#merge(mxj_obj2, mxj_obj1 ,on='用户标识',how='inner')# mxj_obj1和mxj_obj2将用户标识当成重叠列的键合并两个数据集,inner表示取两个数据集的交集.
v3_train=x_train.merge(pd.DataFrame(y_train),on='order_src',how='inner')
#data1_pre['predict_0']=result[:,0]
#将打分结果存储为表
"""
score=[u'order_src',u'apply_time',u'label',u'predict_1']
data_score=data1_pre[score]
df.rename(index=str, columns={"A": "a", "B": "c"})
d1 = pd.cut(data, k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
ma=max(score_v3train[u'predict_1'])
mi=min(score_v3train[u'predict_1'])
#等频率离散化
w = [1.0*i/k for i in range(k+1)]
w = data.describe(percentiles = w)[4:4+k+1] #使用describe函数自动计算分位数
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(data, w, labels = range(k))
"""
#等宽离散化
score_v3train=data1_pre.loc[:,['order_src','apply_time','label','predict_1']]
k=20
seg_score = pd.cut(score_v3train[u'predict_1'], k, labels = range(k)) #等宽离散化，各个类比依次命名为0,1,2,3
score_v3train['seg_score_bin']=seg_score[:]

score_v3_rlr_train=data3_pre.loc[:,['label','predict_1']]
k=20
seg_score = pd.cut(score_v3_rlr_train[u'predict_1'], k, labels = range(k))
score_v3_rlr_train['seg_score_bin']=seg_score[:]
#等频率离散化
k=10
w = [1.0*i/k for i in range(k+1)]
w = score_v3train[u'predict_1'].describe(percentiles = w)[4:4+k+1] #使用describe函数自动计算分位数
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(score_v3train[u'predict_1'], w, labels = range(k))
score_v3train['seg_score_percentile']=d2[:]

w = [1.0*i/k for i in range(k+1)]
w = score_v3_rlr_train[u'predict_1'].describe(percentiles = w)[4:4+k+1]
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(score_v3_rlr_train[u'predict_1'], w, labels = range(k))
score_v3_rlr_train['seg_score_percentile']=d2[:]

bins = [0, 0.5, 1]
percentiles=[0.000017,
0.432947,
0.561455,
0.723682,
0.924376,
0.998325,
0.999667,
0.999988,
1]

label_pre = pd.cut(score_v3train[u'predict_1'], bins)
#pd.cut(ages, bins, right=False)
score_v3train['label_pre']=label_pre[:]
#pd.value_counts(label_pre)

label_pre = pd.cut(score_v3_rlr_train[u'predict_1'], bins)
score_v3_rlr_train['label_pre']=label_pre[:]

grouped = score_v3train.groupby(['seg_score_bin','label'])
grouped['label'].count()
grouped = score_v3train.groupby(['seg_score_percentile','label'])
grouped['label'].count()
grouped = score_v3train.groupby(['label','label_pre'])
grouped['label'].count()
grouped = score_v3_rlr_train.groupby(['label','label_pre'])
grouped['label'].count()
grouped = score_v3_rlr_train.groupby(['seg_score_bin','label'])
grouped['label'].count()
grouped = score_v3_rlr_train.groupby(['seg_score_percentile','label'])
grouped['label'].count()

"""
第一轮0304验证 lkl_card_score.overdue_result_all_new_woe_instant_v3_01val
第一轮0506验证 lkl_card_score.fqz_score_dataset_01val_170506
第二轮0304验证 lkl_card_score.fqz_score_dataset_02val
第二轮0506验证 lkl_card_score.fqz_score_dataset_02val_170506
"""
#第三轮验证样本打分
data_v3test.drop(['ljmx'],axis=1, inplace=True)
data_v3test.replace('\N',0,inplace=True)
#data1.fillna(0) #用0填充无效值NaN
data_v3test1=data_v3test.loc[:,['f_id_cnt',
'f_black_cnt',
'f_pass_cnt',
'f_2_self_tmp_overdue0_ls',
'f_2_self_tmp_overdue3_ls',
'f_2_self_tmp_overdue30_ls',
'f_1_id_cnt',
'f_1_black_cnt',
'f_2_2_tmp_overdue3',
'f_2_2_tmp_overdue30',
'edge_woe_sum',
'edge_woe_max',
'edge_woe_min',
'edge_woe_sum_w',
'depth',
'cnt_x_self',
'cnt_x69_self',
'cnt_x67_self',
'avg_af_score_self',
'avg_af_score_one',
'cnt_x_one_w',
'cnt_x62_one_w',
'cnt_x67_one_w',
'cnt_dt38_one_w',
'cnt_pass_one_w',
'cnt_cur_overdue0_one_w',
'cnt_bankcard_one',
'cnt_logmob_one',
'cnt_recom_one',
'cnt_addbook_one',
'cnt_calllog_one']]
score_v3test=LR_v3_RLR.predict_proba(data_v3test1)
data_v3test1['predict_1']=score_v3test[:,1]

#将打分结果存储为表
score_v3test=data_v3test.loc[:,['order_src','apply_time','label','predict_1']]
seg_score_v3test = pd.cut(score_v3test[u'predict_1'], k, labels = range(k))
score_v3test['seg_score']=seg_score_v3test[:]

score_v3_rlr_test=data3_pre.loc[:,['label','predict_1']]
k=20
seg_score = pd.cut(score_v3_rlr_test[u'predict_1'], k, labels = range(k))
score_v3_rlr_test['seg_score_bin']=seg_score[:]

w = [1.0*i/k for i in range(k+1)]
w = score_v3test[u'predict_1'].describe(percentiles = w)[4:4+k+1] #使用describe函数自动计算分位数
w[0] = w[0]*(1-1e-10)
d3 = pd.cut(score_v3test[u'predict_1'], w, labels = range(k))
score_v3test['seg_score_percentile']=d3[:]
d4 = pd.cut(score_v3test[u'predict_1'], percentiles,labels = range(18))
score_v3test['seg_score_percentile']=d4[:]
d5 = pd.cut(score_v3_rlr_test[u'predict_1'], percentiles,labels = range(8))
score_v3_rlr_test['seg_score_percentile']=d5[:]
score_v3test_0304=score_v3_rlr_test[score_v3_rlr_test[u'apply_time']<='2017-04-30']
score_v3test_0506=score_v3_rlr_test[(score_v3_rlr_test[u'apply_time']>'2017-04-30')&(score_v3_rlr_test[u'apply_time']<='2017-06-30')]

grouped = score_v3test_0304.groupby(['seg_score','label'])
grouped['order_src'].count()
grouped = score_v3test_0506.groupby(['seg_score','label'])
grouped['order_src'].count()
grouped = score_v3test_0304.groupby(['seg_score_percentile','label'])
grouped['order_src'].count()
grouped = score_v3test_0506.groupby(['seg_score_percentile','label'])
grouped['order_src'].count()


"""
from sklearn.metrics import roc_auc_score
score = score_v3test['predict_1']
label = score_v3test['label']
auc = roc_auc_score(label,score)
print 'AUC:',auc  
"""

# 导入0506月验证样本
#第一轮0506月
v1_0506 = r'/data/data3/hive_data/overdue_result_all_new_woe_instant_v3_01val.csv'
v1_0506_dataset = pd.read_csv(v1_0506,delimiter='\001')
v1_0506_dataset.drop(['order_src','apply_time','ljmx'],axis=1, inplace=True)
v1_0506_dataset.replace('\N',0,inplace=True)
v1_0506_cols = v1_0506_dataset.columns[1:]
v1_0506_predict=LR_v1.predict_proba(v1_0506_dataset[	])
v1_0506_dataset['predict_1']=v1_0506_predict[:,1]
#第二轮0606月
v2_0506 = r'/data/data3/hive_data/fqz_score_dataset_02val.csv'
v2_0506_dataset = pd.read_csv(	,delimiter='\001')   
v2_0506_dataset.drop(['order_src','apply_time','ljmx'],axis=1, inplace=True)
v2_0506_dataset.replace('\N',0,inplace=True)
v2_0506_cols = v2_0506_dataset.columns[1:]
v2_0506_predict=LR_v2.predict_proba(v2_0506_dataset[	])
v2_0506_dataset['predict_1']=v2_0506_predict[:,1]

"""
numpy中的ndarray方法和属性:
>>> x=np.arange(10)  #随机生成一个数组，并重新命名一个空间的数组
>>> x.size   #获得数组中元素的个数
>>> x.ndim  #获得数组的维数
>>> x.shape  #获得数组的（行数，列数）

Ndarray对象的方法
ndarray.ptp(axis=None, out=None) : 返回数组的最大值—最小值或者某轴的最大值—最小值
ndarray.clip(a_min, a_max, out=None) : 小于最小值的元素赋值为最小值，大于最大值的元素变为最大值。
ndarray.all()：如果所有元素都为真，那么返回真；否则返回假
ndarray.any()：只要有一个元素为真则返回真
ndarray.swapaxes(axis1, axis2) : 交换两个轴的元素，如下

"""

#用python计算AUC的例子
from sklearn.metrics import roc_auc_score
score = data1_pre['predict_1']
label = data1_pre['label']
auc = roc_auc_score(label,score)
print 'AUC:',auc   ---0.937072308643(全量样本)   0.937374750659（0.7抽样样本）

#画ROC曲线
from sklearn import metrics
#sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
y_true=data1_pre.ix[:,'label']
y_score=data1_pre.ix[:,'predict']     
fpr,tpr,thresholds =sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None)
#fpr=roc[0]
#tpr=roc[1]
#thresholds=roc[2]
import matplotlib.pyplot as plt
#plt.figure(figsize = (8, 4)) #设置图像大小
#plt.plot(x,y,label = '$\sin x+1$', color = 'red', linewidth = 2) #作图，设置标签、线条颜色、线条大小
plt.plot(fpr,tpr,label='$logistic$',color='red')
plt.plot(fpr,fpr,color='blue')
plt.xlabel('false positive rate ') # x轴名称
plt.ylabel('true positive rate') # y轴名称
plt.title('roc curve') #标题
#plt.ylim(0, 2.2) #显示的y轴范围
plt.legend() #显示图例
plt.show() #显示作图结果
data1_pre.to_excel(r"E:\data\py\roc_curve.xlsx") #导出预测结果
#data1_pre.to_csv(r"E:\data\py\roc_curve.csv")
#println("Learned classification forest model:\n" + model.toDebugString) #scala中导出模型结果

###########################################################################   GBDT   ####################################################################
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 0) # 切分为训练集和测试集，验证集比例0.3

"""
from sklearn import ensemble
clf = ensemble.GradientBoostingClassifier()
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 0) # 切分为训练集和测试集，验证集比例0.3
gbdt_model = clf.fit(x_train, y_train) # Training model
predicty_x = gbdt_model.predict_proba(x_test)[:, 1]  # predict: probablity of 1
	
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html	
class sklearn.ensemble.GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, 
subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, 
verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’)
	"""
from sklearn.ensemble import GradientBoostingClassifier
v3_GBDT_Classifier = GradientBoostingClassifier(n_estimators=40,max_depth=6,min_samples_leaf=50)   
v3_GBDT_Classifier.fit(x_train, y_train) 
print(u'model avg right rate is %s' % v3_GBDT_Classifier.score(x_train, y_train)) ---0.908004782607

#保存模型
from sklearn.externals import joblib
joblib.dump(v3_GBDT_Classifier,r"/data/data3/hive_data/GBDT_Classifier_v3.model")
#加载模型
GBDT_v3_C=joblib.load(r"/data/data3/hive_data/GBDT_Classifier_v3.model")
#应用模型进行预测
data1_pre=data1[:] #复制data1
predict_cols = data1_pre.columns[1:]
# 进行预测，并将预测评分存入pre的 predict 列中
result_v3_train_GBDT_C=GBDT_v3_C.predict_proba(data1_pre[predict_cols])
data1_pre['predict_1']=result_v3_train_GBDT_C[:,1]
#data1_pre['predict_0']=result[:,0]
from sklearn.metrics import roc_auc_score
score = data1_pre['predict_1']
label = data1_pre['label']
auc = roc_auc_score(label,score)
print 'AUC:',auc   ---0.95922094172

"""
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
"""
from sklearn.ensemble import GradientBoostingRegressor
v3_GBDT_Regressor = GradientBoostingRegressor(n_estimators=40,max_depth=6,min_samples_leaf=50)   
v3_GBDT_Regressor.fit(x_train, y_train)  
print(u'model avg right rate is %s' % v3_GBDT_Regressor.score(x_train, y_train)) ---0.574295741383

#保存模型
from sklearn.externals import joblib
joblib.dump(v3_GBDT_Regressor,r"/data/data3/hive_data/GBDT_Regressor_v3.model")
#加载模型
GBDT_v3_R=joblib.load(r"/data/data3/hive_data/GBDT_Regressor_v3.model")
#应用模型进行预测
data1_pre=data1[:] #复制data1
predict_cols = data1_pre.columns[1:]
# 进行预测，并将预测评分存入pre的 predict 列中
result_v3_train_GBDT_R=GBDT_v3_R.predict(data1_pre[predict_cols])
data1_pre['predict_1']=result_v3_train_GBDT_R[:]

from sklearn.metrics import roc_auc_score
score = data1_pre['predict_1']
label = data1_pre['label']
auc = roc_auc_score(label,score)
print 'AUC:',auc   ---0.937072308643,0.95922094172,0.959115328559

python /zhuqingwen/python_test.py

"""
连续变量离散化--等频分组
def dataDiscretize(dataSet):  
    m,n = shape(dataSet)    #获取数据集行列（样本数和特征数)  
    disMat = tile([0],shape(dataSet))  #初始化离散化数据集  
    for i in range(n-1):    #由于最后一列为类别，因此遍历前n-1列，即遍历特征列  
        x = [l[i] for l in dataSet] #获取第i+1特征向量  
        y = pd.cut(x,10,labels=[0,1,2,3,4,5,6,7,8,9])   #调用cut函数，将特征离散化为10类，可根据自己需求更改离散化种类  
        for k in range(m):  #将离散化值传入离散化数据集  
            disMat[k][i] = y[k]      
    return disMat  
"""
