import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def read_Data():
    test_file_path = "./Dataset/new_complex_nonlinear_data.xlsx"
    test_data = pd.read_excel(test_file_path)
    x_real = test_data['x_new'].values.reshape(-1, 1)
    y_real = test_data['y_new_complex'].values
    return x_real, y_real

def KNeighbors_CV(X, y, x_real, y_real):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 定义KNN回归模型
    knn_model = KNeighborsRegressor()

    # 定义K值范围
    param_grid = {'n_neighbors': np.arange(1, 21)}  # 尝试K值从1到20

    # 使用网格搜索和交叉验证来寻找最佳的K值
    grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # 输出最佳K值
    print("最佳K值:", grid_search.best_params_)

    # 使用最佳K值的模型进行预测
    best_knn_model = grid_search.best_estimator_

    # 计算均方误差
    y_pred = best_knn_model.predict(x_real)
    mse = mean_squared_error(y_real, y_pred)

    return mse, best_knn_model

def draw_plot(best_gbrt, X, y, x_real, y_real):
    # 在测试集上评估模型
    y_pred = best_gbrt.predict(x_real)

    # 绘制原始数据散点图和拟合曲线
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='Original Data', color='blue')
    plt.scatter(x_real, y_real, label='Test Data', color='red')
    plt.plot(x_real, y_pred, label='Fitted Curve', color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('KNeighborsRegressor')
    plt.legend()
    plt.grid(True)

    plt.savefig("KNeighbor.png")
    plt.show()

if __name__ == '__main__':
    # 读取数据
    data = pd.read_excel("./Dataset/cleaned_data.xlsx")
    X = data.drop(columns=['y_complex'])
    y = data['y_complex']

    # 读取测试集数据
    x_real, y_real = read_Data()

    mse, best_gbrt = KNeighbors_CV(X, y, x_real, y_real)
    print("测试集均方误差(MSE):{}".format(mse))

    draw_plot(best_gbrt, X, y, x_real, y_real)