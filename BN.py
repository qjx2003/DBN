import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# -------------------------------
# 1. 生成合成数据
# -------------------------------
np.random.seed(42)  # 保证结果可复现
N = 1000  # 样本数

# 变量 Smoking：P(Smoking=1)=0.3
smoking = np.random.choice([0, 1], size=N, p=[0.7, 0.3])

# 变量 Cancer：如果吸烟，P(Cancer=1)=0.3；如果不吸烟，P(Cancer=1)=0.05
cancer = np.array([np.random.choice([0, 1], p=[0.7, 0.3]) if s == 1 
                   else np.random.choice([0, 1], p=[0.95, 0.05]) 
                   for s in smoking])

# 变量 Xray：如果患癌，P(Xray=1)=0.9；否则 P(Xray=1)=0.2
xray = np.array([np.random.choice([0, 1], p=[0.1, 0.9]) if c == 1 
                 else np.random.choice([0, 1], p=[0.8, 0.2]) 
                 for c in cancer])

# 变量 Dyspnea（呼吸困难）：如果患癌，P(Dyspnea=1)=0.7；否则 P(Dyspnea=1)=0.1
dyspnea = np.array([np.random.choice([0, 1], p=[0.3, 0.7]) if c == 1 
                    else np.random.choice([0, 1], p=[0.9, 0.1]) 
                    for c in cancer])

# 构造 DataFrame
data = pd.DataFrame({
    'Smoking': smoking,
    'Cancer': cancer,
    'Xray': xray,
    'Dyspnea': dyspnea
})



print("数据预览:")
print(data.head())

data.to_excel("data/BN_sample.xlsx", index  = False)

# -------------------------------
# 2. 构造贝叶斯网络结构
# -------------------------------
# 网络结构：Smoking → Cancer → Xray
#                   │
#                   └→ Dyspnea
model = DiscreteBayesianNetwork([
    ('Smoking', 'Cancer'),
    ('Cancer', 'Xray'),
    ('Cancer', 'Dyspnea')
])

print("\n构造的贝叶斯网络结构（边列表）:")
print(model.edges())

# -------------------------------
# 3. 参数学习：采用 BayesianEstimator 并设置 BDeu 先验
# -------------------------------
model.fit(data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

print("\n学习到的条件概率表（CPDs）：")
for cpd in model.get_cpds():
    print(cpd)

# -------------------------------
# 4. 推断：例如，给定吸烟（Smoking=1），推断患癌（Cancer）的概率分布
# -------------------------------
inference = VariableElimination(model)
posterior = inference.query(variables=['Cancer'], evidence={'Smoking': 1})
print("\n在 Smoking=1 的条件下，Cancer 的推断结果：")
print(posterior)
