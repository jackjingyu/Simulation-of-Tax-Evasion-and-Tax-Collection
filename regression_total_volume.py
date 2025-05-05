import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, f'compare.xlsx')
# 读取Excel数据（替换为你的文件路径）
df = pd.read_excel(file_path)

# 获取不同的cost值
costs = df['cost'].unique()
costs.sort()

# 创建颜色映射
colors = plt.cm.rainbow(np.linspace(0, 1, len(costs)))

# 创建图形
plt.figure(figsize=(12, 7))

# 遍历每个cost值进行拟合
for cost, color in zip(costs, colors):
    # 筛选当前cost的数据
    subset = df[df['cost'] == cost]
    X = subset['audit rate'].values.reshape(-1, 1)
    y = subset['total_volume'].values
    
    # 多项式回归（这里使用二次多项式）
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X)
    
    # 拟合模型
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # 预测值
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_pred = model.predict(X_range_poly)
    
    # 计算R²
    r2 = r2_score(y, model.predict(X_poly))
    
    # 获取回归系数
    coef = model.coef_
    intercept = model.intercept_
    
    # 打印回归方程
    print(f"Cost = {cost}:")
    print(f"prob = {intercept:.6f} + {coef[1]:.6f}*audit_rate")
    print(f"R² = {r2:.6f}\n")
    
    # 绘制原始数据点和拟合曲线
    plt.scatter(X, y, color=color, alpha=0.5, label=f'Data (cost={cost})')
    plt.plot(X_range, y_pred, color=color, 
             label=f'Fit (cost={cost})', linewidth=2)

# 添加图例和标签
plt.title('total_volume vs Inspection success rate', fontsize=14)
plt.xlabel('Inspection success rate', fontsize=12)
plt.ylabel('total_volume', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 显示图形
plt.show()