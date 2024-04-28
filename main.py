import pandas as pd
from DecisionTreeRegressor import DecisionTreeRegressor_grid
from GBDT import GBDT_grid
from KNeighbors import KNeighbors_CV

# 读取数据
data = pd.read_excel("./Dataset/complex_nonlinear_data.xlsx")
X = data.drop(columns=['y_complex'])
y = data['y_complex']

test_file_path = "./Dataset/new_complex_nonlinear_data.xlsx"
test_data = pd.read_excel(test_file_path)
x_real = test_data['x_new'].values.reshape(-1, 1)
y_real = test_data['y_new_complex'].values

mse_diction = { "DecisionTree":0,
                "KNeighbors":0,
                "GBDT":0
               }

for i  in range(0,100):
    a,x = DecisionTreeRegressor_grid(X, y, x_real, y_real)
    mse_diction["DecisionTree"] += a
    a, x = KNeighbors_CV(X, y, x_real, y_real)
    mse_diction["KNeighbors"] += a
    a, x = GBDT_grid(X, y, x_real, y_real)
    mse_diction["GBDT"] += a


print("DecisionTree测试集均方误差(MSE):{}".format(mse_diction["DecisionTree"]*0.01))
print("KNeighbors测试集均方误差(MSE):{}".format(mse_diction["KNeighbors"]*0.01))
print("GBDT测试集均方误差(MSE):{}".format(mse_diction["GBDT"]*0.01))