import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def read_Data():
    test_file_path = "./Dataset/new_complex_nonlinear_data.xlsx"
    test_data = pd.read_excel(test_file_path)
    x_real = test_data['x_new'].values.reshape(-1, 1)
    y_real = test_data['y_new_complex'].values
    return x_real, y_real


def DecisionTreeRegressor_grid(X, y, x_real, y_real):
    # # 设置随机种子
    # random_seed = 42

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 创建决策树回归模型并拟合
    tree_reg = DecisionTreeRegressor()

    # 定义参数网格
    param_grid = {
        # 'max_depth': [None, 2, 5, 100],
        'min_samples_split': [6, 8, 10],
        'min_samples_leaf': [2, 3, 4],
        # 'criterion': ['mse', 'friedman_mse', 'mae'],
        # 'splitter': ['best', 'random'],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    # 网格搜索
    grid_search = GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    '''
    交叉验证折数（CV折数）可以影响模型评估的准确性和稳定性。
    
    准确性影响：CV折数越大，模型在训练数据上的评估结果会更加准确，因为更多的数据被用于训练和验证。但是，CV折数增加也意味着训练时间会增加，因为需要进行更多次的训练和验证。
    
    稳定性影响：CV折数过小可能导致模型评估结果具有较大的方差（variance），即在不同的数据划分上得到的评估结果可能差异较大。增加CV折数可以减少方差，使得评估结果更加稳定。但是，CV折数过大可能会导致模型评估结果的偏差（bias）增加，因为每一次验证集都很小，可能无法很好地代表整个数据集的特征。
    
    因此，在选择CV折数时，需要权衡准确性和稳定性之间的平衡。通常来说，5折或10折交叉验证是比较常用的选择，能够在保证准确性的同时，也保持了一定的稳定性。如果数据量非常大，也可以考虑使用更小的CV折数，例如3折交叉验证，以减少训练时间。
    '''

    # 输出最佳参数
    print("最佳参数:", grid_search.best_params_)

    # 使用最佳参数重新拟合模型
    best_gbrt = grid_search.best_estimator_
    best_gbrt.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = best_gbrt.predict(x_real)
    mse = mean_squared_error(y_real, y_pred)
    # print("测试集均方误差(MSE):", mse)
    # 保存测试集中每个测试点的 MSE 损失值到结果文件
    result_df = pd.DataFrame({'x': x_real.flatten(), 'y_true': y_real, 'y_pred_DT': y_pred})
    result_df['MSE_DT'] = (result_df['y_true'] - result_df['y_pred_DT']) ** 2
    result_file_path = "./Dataset/result.xlsx"
    result_df.to_excel(result_file_path, index=False)
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
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # 读取数据
    data = pd.read_excel("./Dataset/cleaned_data.xlsx")
    X = data.drop(columns=['y_complex'])
    y = data['y_complex']

    # 读取测试集数据
    x_real, y_real = read_Data()

    mse, best_gbrt = DecisionTreeRegressor_grid(X, y, x_real, y_real)
    print("测试集均方误差(MSE):{}".format(mse))

    draw_plot(best_gbrt, X, y, x_real, y_real)
