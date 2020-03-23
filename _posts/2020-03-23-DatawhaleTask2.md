

```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
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



```python
Train_data.head().append(Train_data.tail(3))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SaleID</th>
      <th>name</th>
      <th>regDate</th>
      <th>model</th>
      <th>brand</th>
      <th>bodyType</th>
      <th>fuelType</th>
      <th>gearbox</th>
      <th>power</th>
      <th>kilometer</th>
      <th>...</th>
      <th>v_5</th>
      <th>v_6</th>
      <th>v_7</th>
      <th>v_8</th>
      <th>v_9</th>
      <th>v_10</th>
      <th>v_11</th>
      <th>v_12</th>
      <th>v_13</th>
      <th>v_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>736</td>
      <td>20040402</td>
      <td>30.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60</td>
      <td>12.5</td>
      <td>...</td>
      <td>0.235676</td>
      <td>0.101988</td>
      <td>0.129549</td>
      <td>0.022816</td>
      <td>0.097462</td>
      <td>-2.881803</td>
      <td>2.804097</td>
      <td>-2.420821</td>
      <td>0.795292</td>
      <td>0.914762</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2262</td>
      <td>20030301</td>
      <td>40.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>15.0</td>
      <td>...</td>
      <td>0.264777</td>
      <td>0.121004</td>
      <td>0.135731</td>
      <td>0.026597</td>
      <td>0.020582</td>
      <td>-4.900482</td>
      <td>2.096338</td>
      <td>-1.030483</td>
      <td>-1.722674</td>
      <td>0.245522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>14874</td>
      <td>20040403</td>
      <td>115.0</td>
      <td>15</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>163</td>
      <td>12.5</td>
      <td>...</td>
      <td>0.251410</td>
      <td>0.114912</td>
      <td>0.165147</td>
      <td>0.062173</td>
      <td>0.027075</td>
      <td>-4.846749</td>
      <td>1.803559</td>
      <td>1.565330</td>
      <td>-0.832687</td>
      <td>-0.229963</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>71865</td>
      <td>19960908</td>
      <td>109.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>193</td>
      <td>15.0</td>
      <td>...</td>
      <td>0.274293</td>
      <td>0.110300</td>
      <td>0.121964</td>
      <td>0.033395</td>
      <td>0.000000</td>
      <td>-4.509599</td>
      <td>1.285940</td>
      <td>-0.501868</td>
      <td>-2.438353</td>
      <td>-0.478699</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>111080</td>
      <td>20120103</td>
      <td>110.0</td>
      <td>5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>68</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.228036</td>
      <td>0.073205</td>
      <td>0.091880</td>
      <td>0.078819</td>
      <td>0.121534</td>
      <td>-1.896240</td>
      <td>0.910783</td>
      <td>0.931110</td>
      <td>2.834518</td>
      <td>1.923482</td>
    </tr>
    <tr>
      <th>149997</th>
      <td>149997</td>
      <td>147587</td>
      <td>20101003</td>
      <td>60.0</td>
      <td>11</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.233353</td>
      <td>0.000705</td>
      <td>0.118872</td>
      <td>0.100118</td>
      <td>0.097914</td>
      <td>2.439812</td>
      <td>-1.630677</td>
      <td>2.290197</td>
      <td>1.891922</td>
      <td>0.414931</td>
    </tr>
    <tr>
      <th>149998</th>
      <td>149998</td>
      <td>45907</td>
      <td>20060312</td>
      <td>34.0</td>
      <td>10</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>156</td>
      <td>15.0</td>
      <td>...</td>
      <td>0.256369</td>
      <td>0.000252</td>
      <td>0.081479</td>
      <td>0.083558</td>
      <td>0.081498</td>
      <td>2.075380</td>
      <td>-2.633719</td>
      <td>1.414937</td>
      <td>0.431981</td>
      <td>-1.659014</td>
    </tr>
    <tr>
      <th>149999</th>
      <td>149999</td>
      <td>177672</td>
      <td>19990204</td>
      <td>19.0</td>
      <td>28</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>193</td>
      <td>12.5</td>
      <td>...</td>
      <td>0.284475</td>
      <td>0.000000</td>
      <td>0.040072</td>
      <td>0.062543</td>
      <td>0.025819</td>
      <td>1.978453</td>
      <td>-3.179913</td>
      <td>0.031724</td>
      <td>-1.483350</td>
      <td>-0.342674</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
Train_data.shape
```




    (150000, 31)




```python

Train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 31 columns):
    SaleID               150000 non-null int64
    name                 150000 non-null int64
    regDate              150000 non-null int64
    model                149999 non-null float64
    brand                150000 non-null int64
    bodyType             145494 non-null float64
    fuelType             141320 non-null float64
    gearbox              144019 non-null float64
    power                150000 non-null int64
    kilometer            150000 non-null float64
    notRepairedDamage    150000 non-null object
    regionCode           150000 non-null int64
    seller               150000 non-null int64
    offerType            150000 non-null int64
    creatDate            150000 non-null int64
    price                150000 non-null int64
    v_0                  150000 non-null float64
    v_1                  150000 non-null float64
    v_2                  150000 non-null float64
    v_3                  150000 non-null float64
    v_4                  150000 non-null float64
    v_5                  150000 non-null float64
    v_6                  150000 non-null float64
    v_7                  150000 non-null float64
    v_8                  150000 non-null float64
    v_9                  150000 non-null float64
    v_10                 150000 non-null float64
    v_11                 150000 non-null float64
    v_12                 150000 non-null float64
    v_13                 150000 non-null float64
    v_14                 150000 non-null float64
    dtypes: float64(20), int64(10), object(1)
    memory usage: 35.5+ MB



```python
Train_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SaleID</th>
      <th>name</th>
      <th>regDate</th>
      <th>model</th>
      <th>brand</th>
      <th>bodyType</th>
      <th>fuelType</th>
      <th>gearbox</th>
      <th>power</th>
      <th>kilometer</th>
      <th>...</th>
      <th>v_5</th>
      <th>v_6</th>
      <th>v_7</th>
      <th>v_8</th>
      <th>v_9</th>
      <th>v_10</th>
      <th>v_11</th>
      <th>v_12</th>
      <th>v_13</th>
      <th>v_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>1.500000e+05</td>
      <td>149999.000000</td>
      <td>150000.000000</td>
      <td>145494.000000</td>
      <td>141320.000000</td>
      <td>144019.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>...</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>74999.500000</td>
      <td>68349.172873</td>
      <td>2.003417e+07</td>
      <td>47.129021</td>
      <td>8.052733</td>
      <td>1.792369</td>
      <td>0.375842</td>
      <td>0.224943</td>
      <td>119.316547</td>
      <td>12.597160</td>
      <td>...</td>
      <td>0.248204</td>
      <td>0.044923</td>
      <td>0.124692</td>
      <td>0.058144</td>
      <td>0.061996</td>
      <td>-0.001000</td>
      <td>0.009035</td>
      <td>0.004813</td>
      <td>0.000313</td>
      <td>-0.000688</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43301.414527</td>
      <td>61103.875095</td>
      <td>5.364988e+04</td>
      <td>49.536040</td>
      <td>7.864956</td>
      <td>1.760640</td>
      <td>0.548677</td>
      <td>0.417546</td>
      <td>177.168419</td>
      <td>3.919576</td>
      <td>...</td>
      <td>0.045804</td>
      <td>0.051743</td>
      <td>0.201410</td>
      <td>0.029186</td>
      <td>0.035692</td>
      <td>3.772386</td>
      <td>3.286071</td>
      <td>2.517478</td>
      <td>1.288988</td>
      <td>1.038685</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.991000e+07</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-9.168192</td>
      <td>-5.558207</td>
      <td>-9.639552</td>
      <td>-4.153899</td>
      <td>-6.546556</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>37499.750000</td>
      <td>11156.000000</td>
      <td>1.999091e+07</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>75.000000</td>
      <td>12.500000</td>
      <td>...</td>
      <td>0.243615</td>
      <td>0.000038</td>
      <td>0.062474</td>
      <td>0.035334</td>
      <td>0.033930</td>
      <td>-3.722303</td>
      <td>-1.951543</td>
      <td>-1.871846</td>
      <td>-1.057789</td>
      <td>-0.437034</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>74999.500000</td>
      <td>51638.000000</td>
      <td>2.003091e+07</td>
      <td>30.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>110.000000</td>
      <td>15.000000</td>
      <td>...</td>
      <td>0.257798</td>
      <td>0.000812</td>
      <td>0.095866</td>
      <td>0.057014</td>
      <td>0.058484</td>
      <td>1.624076</td>
      <td>-0.358053</td>
      <td>-0.130753</td>
      <td>-0.036245</td>
      <td>0.141246</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112499.250000</td>
      <td>118841.250000</td>
      <td>2.007111e+07</td>
      <td>66.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>150.000000</td>
      <td>15.000000</td>
      <td>...</td>
      <td>0.265297</td>
      <td>0.102009</td>
      <td>0.125243</td>
      <td>0.079382</td>
      <td>0.087491</td>
      <td>2.844357</td>
      <td>1.255022</td>
      <td>1.776933</td>
      <td>0.942813</td>
      <td>0.680378</td>
    </tr>
    <tr>
      <th>max</th>
      <td>149999.000000</td>
      <td>196812.000000</td>
      <td>2.015121e+07</td>
      <td>247.000000</td>
      <td>39.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>19312.000000</td>
      <td>15.000000</td>
      <td>...</td>
      <td>0.291838</td>
      <td>0.151420</td>
      <td>1.404936</td>
      <td>0.160791</td>
      <td>0.222787</td>
      <td>12.357011</td>
      <td>18.819042</td>
      <td>13.847792</td>
      <td>11.147669</td>
      <td>8.658418</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>



###查看缺失值


```python
Train_data.isnull().sum()
```




    SaleID                  0
    name                    0
    regDate                 0
    model                   1
    brand                   0
    bodyType             4506
    fuelType             8680
    gearbox              5981
    power                   0
    kilometer               0
    notRepairedDamage       0
    regionCode              0
    seller                  0
    offerType               0
    creatDate               0
    price                   0
    v_0                     0
    v_1                     0
    v_2                     0
    v_3                     0
    v_4                     0
    v_5                     0
    v_6                     0
    v_7                     0
    v_8                     0
    v_9                     0
    v_10                    0
    v_11                    0
    v_12                    0
    v_13                    0
    v_14                    0
    dtype: int64




```python
Test_data.isnull().sum()
```




    SaleID                  0
    name                    0
    regDate                 0
    model                   0
    brand                   0
    bodyType             1413
    fuelType             2893
    gearbox              1910
    power                   0
    kilometer               0
    notRepairedDamage       0
    regionCode              0
    seller                  0
    offerType               0
    creatDate               0
    v_0                     0
    v_1                     0
    v_2                     0
    v_3                     0
    v_4                     0
    v_5                     0
    v_6                     0
    v_7                     0
    v_8                     0
    v_9                     0
    v_10                    0
    v_11                    0
    v_12                    0
    v_13                    0
    v_14                    0
    dtype: int64




```python
# nan可视化
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x109724748>




![png](output_10_1.png)



```python
# nan可视化 test
missing = Test_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10966b860>




![png](output_11_1.png)



```python
# 可视化看下缺省值
msno.matrix(Train_data.sample(250))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x103b64da0>




![png](output_12_1.png)



```python
msno.bar(Train_data.sample(1000))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x109916198>




![png](output_13_1.png)



```python
## 2) 查看异常值检测
Train_data['notRepairedDamage'].value_counts()
```




    0.0    111361
    -       24324
    1.0     14315
    Name: notRepairedDamage, dtype: int64




```python
#替换缺失值
Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
Test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
```


```python
#删去对预测没有意义的项
del Train_data["seller"]
del Train_data["offerType"]
del Test_data["seller"]
del Test_data["offerType"]
```

### 2.3.5 了解预测值的分布


```python
import scipy.stats as st
y = Train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    <matplotlib.axes._subplots.AxesSubplot at 0x111621b38>




![png](output_18_2.png)



![png](output_18_3.png)



![png](output_18_4.png)



```python
## 2) 查看skewness and kurtosis
sns.distplot(Train_data['price']);
print("Skewness: %f" % Train_data['price'].skew())
print("Kurtosis: %f" % Train_data['price'].kurt())
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "


    Skewness: 3.346487
    Kurtosis: 18.995183



![png](output_19_2.png)



```python
Train_data.skew(), Train_data.kurt()
```




    (SaleID               6.017846e-17
     name                 5.576058e-01
     regDate              2.849508e-02
     model                1.484388e+00
     brand                1.150760e+00
     bodyType             9.915299e-01
     fuelType             1.595486e+00
     gearbox              1.317514e+00
     power                6.586318e+01
     kilometer           -1.525921e+00
     notRepairedDamage    2.430640e+00
     regionCode           6.888812e-01
     creatDate           -7.901331e+01
     price                3.346487e+00
     v_0                 -1.316712e+00
     v_1                  3.594543e-01
     v_2                  4.842556e+00
     v_3                  1.062920e-01
     v_4                  3.679890e-01
     v_5                 -4.737094e+00
     v_6                  3.680730e-01
     v_7                  5.130233e+00
     v_8                  2.046133e-01
     v_9                  4.195007e-01
     v_10                 2.522046e-02
     v_11                 3.029146e+00
     v_12                 3.653576e-01
     v_13                 2.679152e-01
     v_14                -1.186355e+00
     dtype: float64, SaleID                 -1.200000
     name                   -1.039945
     regDate                -0.697308
     model                   1.740483
     brand                   1.076201
     bodyType                0.206937
     fuelType                5.880049
     gearbox                -0.264161
     power                5733.451054
     kilometer               1.141934
     notRepairedDamage       3.908072
     regionCode             -0.340832
     creatDate            6881.080328
     price                  18.995183
     v_0                     3.993841
     v_1                    -1.753017
     v_2                    23.860591
     v_3                    -0.418006
     v_4                    -0.197295
     v_5                    22.934081
     v_6                    -1.742567
     v_7                    25.845489
     v_8                    -0.636225
     v_9                    -0.321491
     v_10                   -0.577935
     v_11                   12.568731
     v_12                    0.268937
     v_13                   -0.438274
     v_14                    2.393526
     dtype: float64)




```python
sns.distplot(Train_data.skew(),color='blue',axlabel ='Skewness')
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    <matplotlib.axes._subplots.AxesSubplot at 0x10762bba8>




![png](output_21_2.png)



```python
sns.distplot(Train_data.kurt(),color='orange',axlabel ='Kurtness')
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "





    <matplotlib.axes._subplots.AxesSubplot at 0x1a2525a5f8>




![png](output_22_2.png)



```python
## 3) 查看预测值的具体频数
plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()
```


![png](output_23_0.png)



```python
# price 分布不均匀，可以进行log变换进行预测
plt.hist(np.log(Train_data['price']), orientation = 'vertical',histtype = 'bar', color ='red') 
plt.show()
```


![png](output_24_0.png)


2.3.6 特征分为类别特征和数字特征，并对类别特征查看unique分布


```python
# 分离label即预测值
Y_train = Train_data['price']
```


```python
#对自变量进行分类
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]
```


```python
#对特征进行nunique分布
for i in categorical_features:
    print (i+"的特征分布如下")
    print (("{}特征有个{}不同的值".format(i, Train_data[i].nunique())))
    print(Train_data[i].value_counts())
  #  if i != 'communityName':
        #plt.hist(Train_data[i], bins=3)
        #plt.show()

```

    name的特征分布如下
    name特征有个99662不同的值
    708       282
    387       282
    55        280
    1541      263
    203       233
    53        221
    713       217
    290       197
    1186      184
    911       182
    2044      176
    1513      160
    1180      158
    631       157
    893       153
    2765      147
    473       141
    1139      137
    1108      132
    444       129
    306       127
    2866      123
    2402      116
    533       114
    1479      113
    422       113
    4635      110
    725       110
    964       109
    1373      104
             ... 
    89083       1
    95230       1
    164864      1
    173060      1
    179207      1
    181256      1
    185354      1
    25564       1
    19417       1
    189324      1
    162719      1
    191373      1
    193422      1
    136082      1
    140180      1
    144278      1
    146327      1
    148376      1
    158621      1
    1404        1
    15319       1
    46022       1
    64463       1
    976         1
    3025        1
    5074        1
    7123        1
    11221       1
    13270       1
    174485      1
    Name: name, Length: 99662, dtype: int64
    model的特征分布如下
    model特征有个248不同的值
    0.0      11762
    19.0      9573
    4.0       8445
    1.0       6038
    29.0      5186
    48.0      5052
    40.0      4502
    26.0      4496
    8.0       4391
    31.0      3827
    13.0      3762
    17.0      3121
    65.0      2730
    49.0      2608
    46.0      2454
    30.0      2342
    44.0      2195
    5.0       2063
    10.0      2004
    21.0      1872
    73.0      1789
    11.0      1775
    23.0      1696
    22.0      1524
    69.0      1522
    63.0      1469
    7.0       1460
    16.0      1349
    88.0      1309
    66.0      1250
             ...  
    141.0       37
    133.0       35
    216.0       30
    202.0       28
    151.0       26
    226.0       26
    231.0       23
    234.0       23
    233.0       20
    198.0       18
    224.0       18
    227.0       17
    237.0       17
    220.0       16
    230.0       16
    239.0       14
    223.0       13
    236.0       11
    241.0       10
    232.0       10
    229.0       10
    235.0        7
    246.0        7
    243.0        4
    244.0        3
    245.0        2
    209.0        2
    240.0        2
    242.0        2
    247.0        1
    Name: model, Length: 248, dtype: int64
    brand的特征分布如下
    brand特征有个40不同的值
    0     31480
    4     16737
    14    16089
    10    14249
    1     13794
    6     10217
    9      7306
    5      4665
    13     3817
    11     2945
    3      2461
    7      2361
    16     2223
    8      2077
    25     2064
    27     2053
    21     1547
    15     1458
    19     1388
    20     1236
    12     1109
    22     1085
    26      966
    30      940
    17      913
    24      772
    28      649
    32      592
    29      406
    37      333
    2       321
    31      318
    18      316
    36      228
    34      227
    33      218
    23      186
    35      180
    38       65
    39        9
    Name: brand, dtype: int64
    bodyType的特征分布如下
    bodyType特征有个8不同的值
    0.0    41420
    1.0    35272
    2.0    30324
    3.0    13491
    4.0     9609
    5.0     7607
    6.0     6482
    7.0     1289
    Name: bodyType, dtype: int64
    fuelType的特征分布如下
    fuelType特征有个7不同的值
    0.0    91656
    1.0    46991
    2.0     2212
    3.0      262
    4.0      118
    5.0       45
    6.0       36
    Name: fuelType, dtype: int64
    gearbox的特征分布如下
    gearbox特征有个2不同的值
    0.0    111623
    1.0     32396
    Name: gearbox, dtype: int64
    notRepairedDamage的特征分布如下
    notRepairedDamage特征有个2不同的值
    0.0    111361
    1.0     14315
    Name: notRepairedDamage, dtype: int64
    regionCode的特征分布如下
    regionCode特征有个7905不同的值
    419     369
    764     258
    125     137
    176     136
    462     134
    428     132
    24      130
    1184    130
    122     129
    828     126
    70      125
    827     120
    207     118
    1222    117
    2418    117
    85      116
    2615    115
    2222    113
    759     112
    188     111
    1757    110
    1157    109
    2401    107
    1069    107
    3545    107
    424     107
    272     107
    451     106
    450     105
    129     105
           ... 
    6324      1
    7372      1
    7500      1
    8107      1
    2453      1
    7942      1
    5135      1
    6760      1
    8070      1
    7220      1
    8041      1
    8012      1
    5965      1
    823       1
    7401      1
    8106      1
    5224      1
    8117      1
    7507      1
    7989      1
    6505      1
    6377      1
    8042      1
    7763      1
    7786      1
    6414      1
    7063      1
    4239      1
    5931      1
    7267      1
    Name: regionCode, Length: 7905, dtype: int64


### 2.3.7 数字特征分析


```python
numeric_features.append('price')
```


```python
## 1) 相关性分析
price_numeric = Train_data[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending = False),'\n')
```

    price        1.000000
    v_12         0.692823
    v_8          0.685798
    v_0          0.628397
    power        0.219834
    v_5          0.164317
    v_2          0.085322
    v_6          0.068970
    v_1          0.060914
    v_14         0.035911
    v_13        -0.013993
    v_7         -0.053024
    v_4         -0.147085
    v_9         -0.206205
    v_10        -0.246175
    v_11        -0.275320
    kilometer   -0.440519
    v_3         -0.730946
    Name: price, dtype: float64 
    



```python
f , ax = plt.subplots(figsize = (7, 7))

plt.title('Correlation of Numeric Features with Price',y=1,size=16)

sns.heatmap(correlation,square = True,  vmax=0.8)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x107687080>




![png](output_32_1.png)



```python
del price_numeric['price']
```


```python
## 2) 查看几个特征得 偏度和峰值
for col in numeric_features:
    print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(Train_data[col].skew()) , 
          '   ' ,
          'Kurtosis: {:06.2f}'.format(Train_data[col].kurt())  
         )
```

    power           Skewness: 65.86     Kurtosis: 5733.45
    kilometer       Skewness: -1.53     Kurtosis: 001.14
    v_0             Skewness: -1.32     Kurtosis: 003.99
    v_1             Skewness: 00.36     Kurtosis: -01.75
    v_2             Skewness: 04.84     Kurtosis: 023.86
    v_3             Skewness: 00.11     Kurtosis: -00.42
    v_4             Skewness: 00.37     Kurtosis: -00.20
    v_5             Skewness: -4.74     Kurtosis: 022.93
    v_6             Skewness: 00.37     Kurtosis: -01.74
    v_7             Skewness: 05.13     Kurtosis: 025.85
    v_8             Skewness: 00.20     Kurtosis: -00.64
    v_9             Skewness: 00.42     Kurtosis: -00.32
    v_10            Skewness: 00.03     Kurtosis: -00.58
    v_11            Skewness: 03.03     Kurtosis: 012.57
    v_12            Skewness: 00.37     Kurtosis: 000.27
    v_13            Skewness: 00.27     Kurtosis: -00.44
    v_14            Skewness: -1.19     Kurtosis: 002.39
    price           Skewness: 03.35     Kurtosis: 019.00



```python
## 3) 每个数字特征得分布可视化
f = pd.melt(Train_data, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    /anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](output_35_1.png)



```python
## 4) 数字特征相互之间的关系可视化
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
```


![png](output_36_0.png)



```python
## 5) 多变量互相回归关系可视化
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2, figsize=(24, 20))
# ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
v_12_scatter_plot = pd.concat([Y_train,Train_data['v_12']],axis = 1)
sns.regplot(x='v_12',y = 'price', data = v_12_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

v_8_scatter_plot = pd.concat([Y_train,Train_data['v_8']],axis = 1)
sns.regplot(x='v_8',y = 'price',data = v_8_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

v_0_scatter_plot = pd.concat([Y_train,Train_data['v_0']],axis = 1)
sns.regplot(x='v_0',y = 'price',data = v_0_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

power_scatter_plot = pd.concat([Y_train,Train_data['power']],axis = 1)
sns.regplot(x='power',y = 'price',data = power_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

v_5_scatter_plot = pd.concat([Y_train,Train_data['v_5']],axis = 1)
sns.regplot(x='v_5',y = 'price',data = v_5_scatter_plot,scatter= True, fit_reg=True, ax=ax5)

v_2_scatter_plot = pd.concat([Y_train,Train_data['v_2']],axis = 1)
sns.regplot(x='v_2',y = 'price',data = v_2_scatter_plot,scatter= True, fit_reg=True, ax=ax6)

v_6_scatter_plot = pd.concat([Y_train,Train_data['v_6']],axis = 1)
sns.regplot(x='v_6',y = 'price',data = v_6_scatter_plot,scatter= True, fit_reg=True, ax=ax7)

v_1_scatter_plot = pd.concat([Y_train,Train_data['v_1']],axis = 1)
sns.regplot(x='v_1',y = 'price',data = v_1_scatter_plot,scatter= True, fit_reg=True, ax=ax8)

v_14_scatter_plot = pd.concat([Y_train,Train_data['v_14']],axis = 1)
sns.regplot(x='v_14',y = 'price',data = v_14_scatter_plot,scatter= True, fit_reg=True, ax=ax9)

v_13_scatter_plot = pd.concat([Y_train,Train_data['v_13']],axis = 1)
sns.regplot(x='v_13',y = 'price',data = v_13_scatter_plot,scatter= True, fit_reg=True, ax=ax10)


```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a21c80208>




![png](output_37_1.png)



```python
## 2) 类别特征箱形图可视化

# 因为 name和 regionCode的类别太稀疏了，这里我们把不稀疏的几类画一下
categorical_features = ['model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'notRepairedDamage']
for c in categorical_features:
    Train_data[c] = Train_data[c].astype('category')
    if Train_data[c].isnull().any():
        Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
        Train_data[c] = Train_data[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")
```


![png](output_38_0.png)



```python
## 3) 类别特征的小提琴图可视化
catg_list = categorical_features
target = 'price'
for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=Train_data)
    plt.show()
```


![png](output_39_0.png)



![png](output_39_1.png)



![png](output_39_2.png)



![png](output_39_3.png)



![png](output_39_4.png)



![png](output_39_5.png)



```python
## 4) 类别特征的柱形图可视化
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")
```


![png](output_40_0.png)



```python
##  5) 类别特征的每个类别频数可视化(count_plot)
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data,  value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")


```


![png](output_41_0.png)

