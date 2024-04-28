import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def read_Data () :
    test_file_path = "./Dataset/new_complex_nonlinear_data.xlsx"
    test_data = pd.read_excel(test_file_path)
    x_real = test_data['x_new'].values.reshape(-1, 1)
    y_real = test_data['y_new_complex'].values
    return x_real, y_real

def GBDT_grid (x, y, x_real, y_real):
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 60, 65],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [1, 2, 3]
    }

    # 创建梯度提升回归模型
    gbrt = GradientBoostingRegressor()

    # 网格搜索
    grid_search = GridSearchCV(gbrt, param_grid, cv=5, scoring='r2')
    grid_search.fit(x_train, y_train)

    # 输出最佳参数
    print("Best parameters:", grid_search.best_params_)

    # 使用最佳参数重新拟合模型
    best_gbrt = grid_search.best_estimator_
    best_gbrt.fit(x_train, y_train)

    # 在测试集上评估模型
    y_pred = best_gbrt.predict(x_real)
    mse = mean_squared_error(y_real, y_pred)
    # print("测试集均方误差(MSE):", mse)
    return mse, best_gbrt
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
    plt.title('GradientBoostingDecisionTree Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存测试集中每个测试点的 MSE 损失值到结果文件
    result_df = pd.DataFrame({'x': x_real.flatten(), 'y_true': y_real, 'y_pred': y_pred})
    result_df['MSE'] = (result_df['y_true'] - result_df['y_pred'])**2
    result_file_path = "./Dataset/result.xlsx"
    result_df.to_excel(result_file_path, index=False)

if __name__ == '__main__' :
    # 读取数据
    data = pd.read_excel("./Dataset/cleaned_data.xlsx")
    X = data.drop(columns=['y_complex'])
    y = data['y_complex']

    # 读取测试集数据
    x_real, y_real = read_Data()

    mse, best_gbrt = GBDT_grid(X, y, x_real, y_real)
    print("测试集均方误差(MSE):{}".format(mse))

    draw_plot(best_gbrt, X, y, x_real, y_real)
