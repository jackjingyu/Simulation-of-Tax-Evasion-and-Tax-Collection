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

# 获取唯一的cost值并排序
costs = sorted(df['cost'].unique())
# 使用rainbow颜色映射
colors = plt.cm.rainbow(np.linspace(0, 1, len(costs)))
linestyles = ['-', '--', ':']  # 分别对应good/bad/neutral volume

# 创建图形
plt.figure(figsize=(14, 8))

# 遍历每种volume类型
for vol_type, linestyle in zip(['good_asset', 'bad_asset', 'neutral_asset'], linestyles):
    # 创建子图（可选：如果希望分开显示）
    # plt.figure(figsize=(10, 6))
    
    # 遍历每个cost值
    for i, cost in enumerate(costs):
        # 筛选数据
        subset = df[df['cost'] == cost]
        X = subset['audit rate'].values.reshape(-1, 1)
        y = subset[vol_type].values
        
        # 多项式回归（二次）
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
        print(f"{vol_type} (cost={cost}):")
        print(f"y = {intercept:.3f} + {coef[1]:.3f}*x")
        print(f"R² = {r2:.3f}\n")
        
        # 绘制原始数据点和拟合曲线
        plt.scatter(X, y, color=colors[i], alpha=0.5)
        plt.plot(X_range, y_pred, 
                 label=f'{vol_type} (cost={cost})', 
                 color=colors[i], 
                 linestyle=linestyle)

# 添加图例和标签
plt.title('Asset vs Inspection success rate')
plt.xlabel('Inspection success rate')
plt.ylabel('Asset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()