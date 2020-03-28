---
layout: post
title: Datawhale Task3 学习笔记
categories: Python
description: 交作业交作业
keywords: 
---
**数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。那特征工程到底是什么呢？顾名思义，其本质是一项工程活动，目的是最大限度地从原始数据中提取特征以供算法和模型使用。**

常见的特征工程包括：
1. 异常处理：
    - 通过箱线图（或 3-Sigma）分析删除异常值；
    - BOX-COX 转换（处理有偏分布）；
    - 长尾截断；
2. 特征归一化/标准化：
    - 标准化（转换为标准正态分布）；
    - 归一化（抓换到 [0,1] 区间）；
    - 针对幂律分布，可以采用公式： $log(\frac{1+x}{1+median})$
3. 数据分桶：
    - 等频分桶；
    - 等距分桶；
    - Best-KS 分桶（类似利用基尼指数进行二分类）；
    - 卡方分桶；
4. 缺失值处理：
    - 不处理（针对类似 XGBoost 等树模型）；
    - 删除（缺失数据太多）；
    - 插值补全，包括均值/中位数/众数/建模预测/多重插补/压缩感知补全/矩阵补全等；
    - 分箱，缺失值一个箱；
5. 特征构造：
    - 构造统计量特征，报告计数、求和、比例、标准差等；
    - 时间特征，包括相对时间和绝对时间，节假日，双休日等；
    - 地理信息，包括分箱，分布编码等方法；
    - 非线性变换，包括 log/ 平方/ 根号等；
    - 特征组合，特征交叉；
    - 仁者见仁，智者见智。
6. 特征筛选
    - 过滤式（filter）：先对数据进行特征选择，然后在训练学习器，常见的方法有 Relief/方差选择发/相关系数法/卡方检验法/互信息法；
    - 包裹式（wrapper）：直接把最终将要使用的学习器的性能作为特征子集的评价准则，常见方法有 LVM（Las Vegas Wrapper） ；
    - 嵌入式（embedding）：结合过滤式和包裹式，学习器训练过程中自动进行了特征选择，常见的有 lasso 回归；
7. 降维
    - PCA/ LDA/ ICA；
    - 特征选择也是一种降维。


```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter

%matplotlib inline

from sklearn.preprocessing import StandardScaler
```


```python
Train_data = pd.read_csv('/Users/yangjingchi/Desktop/data/train123.csv', sep=" ")
Test_data = pd.read_csv('/Users/yangjingchi/Desktop/data/testA.csv', sep=" ")
```


```python
print('Train data shape:',Train_data.shape)
print('Test data shape:',Test_data.shape)
```

    Train data shape: (150000, 31)
    Test data shape: (50000, 30)


1. 异常处理：

#3σ原则：如果数据服从正态分布，异常值被定义为一组测定值中与平均值的偏差超过3倍的值# 本案例缺失值比较多的三项不适用该原则进行处理

我们从价格开始分析，通过画图发现价格分布很不均匀，需要做归一化。


```python
import scipy.stats as st
y = Train_data['price']
sns.kdeplot(y, shade=True)
sns.kdeplot(y, bw=.2, label="bw: 0.2")
sns.kdeplot(y, bw=2, label="bw: 2")
plt.legend()
```




    <matplotlib.legend.Legend at 0x11a66e358>




![png](output_7_1.png)



```python
y.plot.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11a8e5e80>




![png](output_8_1.png)



```python
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
y = np.log(y + 1) 
y= ((y - np.min(y)) / (np.max(y) - np.min(y)))
y.plot.hist()

#归一化之后的价格好了很多
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c3fef98>




![png](output_9_1.png)



