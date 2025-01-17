import pandas as pd
import numpy as np
a=pd.read_csv('truedata.csv')
b=pd.read_csv('uadata.csv')
c=pd.read_csv('lstm-data.csv')
d=pd.read_csv('lstmv2c-data.csv')
true=a.iloc[:, 2:42]
enkf=b.iloc[:, 2:42]
gru=c.iloc[:, 2:42]
gruv2=d.iloc[:, 2:42]

import numpy as np
#############################################相关系数计算##########
# 创建两个随机的2001x40维的数据矩阵
data1 = true
data2 = enkf
# 将矩阵展平为一维数组
data1_flattened = data1.values.flatten()
data2_flattened = data2.values.flatten()
# 计算两个一维数组之间的相关系数
correlation_coefficient = np.corrcoef(data1_flattened, data2_flattened)[0, 1]
print("相关系数:", correlation_coefficient)
data1 = true
data2 = gru
# 将矩阵展平为一维数组
data1_flattened = data1.values.flatten()
data2_flattened = data2.values.flatten()
# 计算两个一维数组之间的相关系数
correlation_coefficient = np.corrcoef(data1_flattened, data2_flattened)[0, 1]
print("相关系数:", correlation_coefficient)

#############################################均方误差计算##########
import numpy as np
# 生成两个2001x40维的随机矩阵
a = true
b = enkf
# 计算两个矩阵之间的均方误差
mse = np.mean((a - b) ** 2)
print("均方误差:", mse)
# 计算RMSE
rmse = np.sqrt(mse)
print("均方根误差(RMSE):", rmse)
a = true
b = gru
# 计算两个矩阵之间的均方误差
mse = np.mean((a - b) ** 2)
print("均方误差:", mse)
# 计算RMSE
rmse = np.sqrt(mse)
print("均方根误差(RMSE):", rmse)
#############################################平均绝对误差计算##########
data1 = true
data2 = enkf
# 计算两个矩阵之间的绝对差异
absolute_differences = np.abs(data1 - data2)
# 计算所有元素的平均绝对误差
mae = np.mean(absolute_differences)
print("平均绝对误差 (MAE):", mae)
data1 = true
data2 = gru
# 计算两个矩阵之间的绝对差异
absolute_differences = np.abs(data1 - data2)
# 计算所有元素的平均绝对误差
mae = np.mean(absolute_differences)
print("平均绝对误差 (MAE):", mae)
#############################################偏差计算##########
data1 = true
data2 = enkf
# 计算两个矩阵之间的差值
differences = data2 - data1  # 通常预测值 - 实际值
# 计算偏差，即所有元素差值的平均值
bias = np.mean(differences)
print("偏差 (Bias):", bias)
data1 = true
data2 = gru
# 计算两个矩阵之间的差值
differences = data2 - data1  # 通常预测值 - 实际值
# 计算偏差，即所有元素差值的平均值
bias = np.mean(differences)
print("偏差 (Bias):", bias)
#############################################效率系数计算##########
data1 = true
data2 = enkf
# 将矩阵拉平为一维数组
observed = data1.values.flatten()
predicted = data2.values.flatten()
# 计算实际观测值的平均值
mean_observed = np.mean(observed)
# 计算分子（模型误差）
numerator = np.sum((observed - predicted) ** 2)
# 计算分母（方差总量）
denominator = np.sum((observed - mean_observed) ** 2)
# 计算NSE
NSE = 1 - (numerator / denominator)
print("纳什-苏特克利夫效率系数 (NSE):", NSE)
data1 = true
data2 = gru
# 将矩阵拉平为一维数组
observed = data1.values.flatten()
predicted = data2.values.flatten()
# 计算实际观测值的平均值
mean_observed = np.mean(observed)
# 计算分子（模型误差）
numerator = np.sum((observed - predicted) ** 2)
# 计算分母（方差总量）
denominator = np.sum((observed - mean_observed) ** 2)
# 计算NSE
NSE = 1 - (numerator / denominator)
print("纳什-苏特克利夫效率系数 (NSE):", NSE)