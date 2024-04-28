import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r".\Dataset\complex_nonlinear_data.xlsx"
data = pd.read_excel(file_path)

# 提取x和y值
x = data['x'].values.reshape(-1, 1)
y = data['y_complex'].values.reshape(-1, 1)

# 创建并训练孤立森林模型
model = IsolationForest(contamination=0.02)  # 设置异常值比例为2%
model.fit(y)

# 预测异常值
outliers = model.predict(y)
outliers_indices = [i for i, val in enumerate(outliers) if val == -1]

# 打印异常值坐标
print("异常值坐标:")
for i in outliers_indices:
    print(f"x: {x[i][0]}, y: {y[i][0]}")

# 绘制散点图
plt.scatter(x, y, color='blue', label='Normal')
plt.scatter(x[outliers_indices], y[outliers_indices], color='red', label='Abnormal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('check result')
plt.legend()
plt.show()

# 去除异常值
cleaned_data = data.drop(index=outliers_indices)

# 将处理后的数据写入新的 Excel 文件
output_file_path = r".\Dataset\cleaned_data.xlsx"
cleaned_data.to_excel(output_file_path, index=False)