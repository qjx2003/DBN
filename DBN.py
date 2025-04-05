import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import glob
import os
import ast 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. 生成合成数据
# -------------------------------
# 样本数
N = 1000

# 时间切片0：生成变量 A0 和 B0
# A0 ~ Bernoulli(0.6)
A0 = np.random.choice([0, 1], size=N, p=[0.4, 0.6])
# B0 依赖于 A0：若 A0==0，则 P(B0=1)=0.3；若 A0==1，则 P(B0=1)=0.8
B0 = np.array([np.random.choice([0, 1], p=[0.7, 0.3]) if a == 0 
               else np.random.choice([0, 1], p=[0.2, 0.8]) for a in A0])
data_t0 = pd.DataFrame({'A': A0, 'B': B0})

# 时间切片1：生成变量 A1 和 B1
# A1 依赖于 A0：若 A0==0，则 P(A1=1)=0.4；若 A0==1，则 P(A1=1)=0.7
A1 = np.array([np.random.choice([0,1], p=[0.6, 0.4]) if a0==0 
               else np.random.choice([0,1], p=[0.3, 0.7]) for a0 in A0])
# B1 依赖于 B0：若 B0==0，则 P(B1=1)=0.35；若 B0==1，则 P(B1=1)=0.75
B1 = np.array([np.random.choice([0,1], p=[0.65, 0.35]) if b0==0 
               else np.random.choice([0,1], p=[0.25, 0.75]) for b0 in B0])
data_t1 = pd.DataFrame({'A': A1, 'B': B1})

print("时间切片 0 数据预览:")
print(data_t0.head())
print("\n时间切片 1 数据预览:")
print(data_t1.head())

# -------------------------------
# 2. 构造动态贝叶斯网络（DBN）结构
# -------------------------------
dbn = DBN()

# 添加两个时间切片中所有变量的节点
for var in ['A', 'B']:
    dbn.add_node((var, 0))
    dbn.add_node((var, 1))

# 添加时间切片内的边（intra-slice）
# 时间 0：A0 -> B0
dbn.add_edge(('A', 0), ('B', 0))
# 时间 1：A1 -> B1
dbn.add_edge(('A', 1), ('B', 1))

# 添加跨时间切片的边（inter-slice）
# 自回归边：A0 -> A1, B0 -> B1
dbn.add_edge(('A', 0), ('A', 1))
dbn.add_edge(('B', 0), ('B', 1))
# 可选跨切片边：A0 -> B1（可以捕捉跨变量的时间依赖）
dbn.add_edge(('A', 0), ('B', 1))

print("\n构造的 DBN 结构边：")
print(list(dbn.edges()))

# -------------------------------
# 3. 参数学习：使用贝叶斯估计器（结合先验分布）
# -------------------------------
# 这里使用 BayesianEstimator，指定 prior_type 为 'BDeu'（Dirichlet 先验），并设置 equivalent_sample_size（先验样本量）
dbn.fit([data_t0, data_t1], estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

# 输出学习到的条件概率分布（CPDs）
print("\n学习到的条件概率分布：")
for cpd in dbn.get_cpds():
    print(cpd)
