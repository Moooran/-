# 多种回归分析模型对比
小模型数据集下进行回归分析，并对各个算法进行性能分析

# 文件说明
IsolationForest文件：用于数据处理
Data_fitting文件：包含插值法、局部法、多项式法
main文件：运行后决策树法、k邻近法、GBDT法将循环执行100次并计算mse
DecisionTree、KNeighbors、GBDT文件：均可单独执行，执行后会反馈最佳参数、mse以及绘制图片
