import pandas as pd
import numpy as np
import copy

data = pd.DataFrame(pd.read_excel('华北.xlsx'))

label_need = data.keys()[1:]
# print(label_need)
data1 = data[label_need].values
# print(data1)
diqu = data['地区'].values
# print(diqu)
data2 = data1
# print(data2)

[m, n] = data2.shape
data3 = copy.deepcopy(data2)
# print(data3)

# 归一化0.002~1
ymin = 0.002
ymax = 1
for j in range(n):
    d_max = max(data2[:,j])
    d_min = min(data2[:,j])
    data3[:,j] = (ymax-ymin)*(data2[:,j]-d_min)/(d_max-d_min)+ymin
# print(data3)

# 信息熵
p = copy.deepcopy(data3)
for j in range(n):
    p[:,j] = data3[:,j]/sum(data3[:,j])
# print(p)
E = copy.deepcopy(data3[0, :])
for j in range(n):
    E[j] = -1/np.log(m)*sum(p[:, j]*np.log(p[:,j]))
# print(E)

# 计算权重
W = (1 - E)/sum(1 - E)
# print(W)

s = np.dot(data3,W)
score = 100*s/max(s)
# for i in range(len(score)):
#     print(f"{diqu[i]}得分为：{score[i]}")

for i in score:
    print(i)
