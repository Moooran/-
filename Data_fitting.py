import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_squared_error

# 定义计算MSE的函数
def calculate_mse(regressor_name, regressor_func, x_real, y_real):
    y_pred = regressor_func(x_real)
    mse = mean_squared_error(y_real, y_pred)
    print("{}测试集均方误差(MSE):{}".format(regressor_name, mse))
    return mse

# 读取数据
data = pd.read_excel("./Dataset/cleaned_data.xlsx")
x = data['x'].values.reshape(-1)  # 转换为一维数组
y = data['y_complex'].values

# 多项式拟合
# Polynomial regression（多项式回归）：这是一种回归分析方法，它使用多项式函数来拟合数据点，以预测自变量与因变量之间的关系。在这种情况下，使用了三次多项式来拟合数据。
coefficients = np.polyfit(x, y, 18)#尝试3-500
poly_function = np.poly1d(coefficients)

# 局部加权线性回归
# Lowess smoothing（局部加权散点平滑）：这是一种非参数的统计方法，用于平滑数据并识别数据中的趋势。它通过在每个数据点周围拟合局部加权的线性回归模型来实现，以获得数据的平滑估计值。
smoothed = lowess(y, x, frac=0.07)#已尝试0.01-0.5
x_lowess = smoothed[:, 0]
y_lowess = smoothed[:, 1]

# 样条插值
# Cubic spline interpolation（三次样条插值）：这是一种使用三次多项式来逼近一组数据点的方法。它通过在相邻数据点之间拟合三次多项式来创建平滑的曲线。
f_interp1d = interp1d(x, y, kind='cubic')

# 加载测试数据
test_file_path = "./Dataset/new_complex_nonlinear_data.xlsx"
test_data = pd.read_excel(test_file_path)
x_real = test_data['x_new'].values.reshape(-1, 1)
y_real = test_data['y_new_complex'].values.reshape(-1, 1)

# 计算各模型的MSE
calculate_mse("Cubic spline interpolation", f_interp1d, x_real, y_real)
calculate_mse("Polynomial regression", poly_function, x_real, y_real)
calculate_mse("Lowess smoothing", lambda x: np.interp(x, x_lowess, y_lowess), x_real, y_real)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data', color='blue')
plt.scatter(x_real, y_real, label='Test Data', color='red')

# 样条插值
x_interp1d = np.linspace(x.min(), x.max(), 100)
y_interp1d = f_interp1d(x_interp1d)
plt.plot(x_interp1d, y_interp1d, color='green', label='Cubic spline interpolation')

# 多项式拟合
y_poly = poly_function(x_interp1d)
plt.plot(x_interp1d, y_poly, color='purple', label='Polynomial regression')

# 局部加权线性回归
plt.plot(x_lowess, y_lowess, color='orange', label='Lowess smoothing')

plt.xlabel('x')
plt.ylabel('y_complex')
plt.title('Fitting Methods Comparison')
plt.legend()
plt.grid(True)
plt.show()
