import pandas as pd
import numpy as np
import math
from matplotlib.ticker import MaxNLocator
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import random
import warnings
from time import time

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)


class GA(object):
    def __init__(self, x, y, model, purpose, population_size, iter_num, pc, pm,
                 params=None, params_bound=None, cv=5, filter=None):

        # 模型数据集
        self.x = x
        self.y = y

        # 机器学习模型参数 #
        # 确定机器学习模型, KNN或者是MLP
        self.model = model

        # GA用于机器学习的功能：调参Parameter还是特征选择Feature
        self.purpose = purpose

        # 如果是Feature, 则染色体不需要解码，可用来描述特征选择
        # 用作特征选择的算法采取的默认参数
        self.params = params

        # GA特征筛选之前是否进行过滤法

        # 过滤法可选F value或者Mutual Information
        self.filter = filter  # filter=F or filter = MI

        # 过滤法的上下界
        # 超过上界的全部选取 (>= Q3)
        # 低于下界的全部舍弃 (<= Q1)

        # 如果是Parameter
        self.params_bound = params_bound  # 需指明参数的上下界

        # 机器学习performance evaluation的cv
        self.cv = cv

        # GA代码参数 #

        # 种群数量 #
        self.population_size = population_size

        # 遗传代数限制
        self.iter_num = iter_num

        # 交叉和变异概率
        self.pc = pc  # 交叉概率crossover
        self.pm = pm  # 变异概率mutation

    # 初始化种群
    def population_generate(self):

        if self.purpose == 'Parameter':  # GA用作调参

            # 构建空数组准备存储种群
            # 确定参数的染色体长度
            # 如果是整数，根据区间数量计算
            chromosome_length = np.arange(len(self.params_bound))

            # 参数个数
            for i in range(len(chromosome_length)):
                tmp = math.ceil(math.log2(list(self.params_bound.values())[i][1] - list(self.params_bound.values())[i][0] + 1))
                chromosome_length[i] = tmp

            population = np.empty(shape=(self.population_size, np.sum(chromosome_length))).astype(int)

            # 构建种群
            for i in range(self.population_size):
                random_chromosome = np.random.randint(2, size=sum(chromosome_length))
                population[i, ::] = random_chromosome

            return chromosome_length, population.astype(int)

        elif self.purpose == 'Feature':  # GA用作特征选择

            feature_num = self.x.shape[1]

            if self.filter == 'F' or self.filter == 'MI':

                # 如果在GA调参之前采取过滤法
                if self.filter == 'F':

                    # 计算所有X与Y的 F score
                    correlation, _ = f_regression(self.x, self.y)

                elif self.filter == 'MI':

                    # 计算所有X与Y的 MI值
                    correlation = mutual_info_regression(self.x, self.y)

                correlation /= np.max(correlation)

                # 找出小于filter_lower的索引,直接删去
                lower_features = np.where(correlation <= np.percentile(correlation, 25))[0]  # 特征序号

                # 找出大于filter_upper的索引，直接保留
                upper_features = np.where(correlation >= np.percentile(correlation, 75))[0]  # 特征序号

                # 剩余feature的索引
                ga_features = np.delete(np.arange(0, feature_num), np.append(lower_features, upper_features))

                chromosome_length = len(ga_features)

                population = np.empty(shape=(self.population_size, chromosome_length))

                for i in range(self.population_size):
                    # 针对低于filter_upper和高于filter_lower的feature进行GA启发式选择
                    left_feature_index = np.random.randint(2, size=chromosome_length)
                    population[i, ::] = left_feature_index

                return upper_features, ga_features, population.astype(int)

            else:
                # 不使用过滤法
                chromosome_length = feature_num
                population = np.empty(shape=(self.population_size, chromosome_length))

                for i in range(self.population_size):
                    all_feature_index = np.random.randint(2, size=chromosome_length)
                    population[i, ::] = all_feature_index

                return population.astype(int)

        elif self.purpose == 'Parameter+Feature':


            # 针对调参的种群构建
            parameter_chromosome_length = np.arange(len(self.params_bound))
            for i in range(len(self.params_bound)):
                tmp = math.ceil(
                    math.log2(list(self.params_bound.values())[i][1] - list(self.params_bound.values())[i][0] + 1))
                parameter_chromosome_length[i] = tmp
            total_chromosome_length = np.sum(parameter_chromosome_length)
            parameter_population = np.empty(shape=(self.population_size, total_chromosome_length))
            # 构建种群
            for i in range(self.population_size):
                random_chromosome = np.random.randint(2, size=total_chromosome_length)
                parameter_population[i, ::] = random_chromosome


            # 针对特征选择的种群构建
            feature_num = self.x.shape[1]

            # 使用过滤法
            if self.filter is not None:
                if self.filter == 'F':
                    correlation, _ = f_regression(self.x, self.y)
                elif self.filter == 'MI':
                    correlation = mutual_info_regression(self.x, self.y)
                correlation /= np.max(correlation)
                lower_features = np.where(correlation <= np.percentile(correlation, 25))[0]  # 特征序号
                upper_features = np.where(correlation >= np.percentile(correlation, 75))[0]  # 特征序号
                ga_features = np.delete(np.arange(0, feature_num), np.append(lower_features, upper_features))

                feature_chromosome_length = len(ga_features)
                feature_population = np.empty(shape=(self.population_size, feature_chromosome_length))
                for i in range(self.population_size):
                    left_feature_index = np.random.randint(2, size=feature_chromosome_length)
                    feature_population[i, ::] = left_feature_index

                # 合并种群
                population = np.append(parameter_population, feature_population, axis=1).astype(int)

                return parameter_chromosome_length, upper_features, ga_features, feature_population, population

            # 不使用过滤法
            else:
                feature_chromosome_length = feature_num
                feature_population = np.empty(shape=(self.population_size, feature_chromosome_length))
                for i in range(self.population_size):
                    left_feature_index = np.random.randint(2, size=feature_chromosome_length)
                    feature_population[i, ::] = left_feature_index

                population = np.append(parameter_population, feature_population, axis=1).astype(int)

                return parameter_chromosome_length, feature_chromosome_length, feature_population, population

    # 解码并得到参数或者待选择的特征
    def decoding(self, population):

        if self.purpose == 'Parameter':  # GA用作调参

            chromosome_length = self.population_generate()[0]

            population_value = np.empty(shape=(self.population_size, len(self.params_bound)))

            # 记录参数值
            for i in range(self.population_size):  # 循环种群每个个体

                copied_population = population[i, ::].copy()

                for j, num in enumerate(chromosome_length):  # 循环一个种群个体的每个参数染色体

                    new_chromosome = copied_population[:num]

                    strs = ''.join(str(int(k)) for k in new_chromosome)

                    int_value = int(strs, 2)

                    # 参数值
                    upper_bound = list(self.params_bound.values())[j][1]
                    lower_bound = list(self.params_bound.values())[j][0]

                    para_value = round(lower_bound + int_value * (upper_bound - lower_bound) / (pow(2, num) - 1))

                    population_value[i, j] = para_value

                    copied_population = np.delete(copied_population, range(num))

            return population_value

        elif self.purpose == 'Feature':  # GA用作特征选择

            population_value = []

            # 采用过滤法
            if self.filter == 'F' or self.filter == 'MI':

                # 必须选择的feature (>Q3)
                must_select_features = self.population_generate()[0]

                # GA选择的features
                selected_ga_features = self.population_generate()[1]

                for i in range(self.population_size):

                    selected_features = np.append(must_select_features,
                                                      selected_ga_features[population[i, ::] == 1])

                    population_value.append(selected_features)

            # 不使用过滤法
            else:
                chromosome_length = population.shape[1]

                for i in range(self.population_size):
                    selected_features = np.arange(chromosome_length)[population[i, ::] == 1]
                    population_value.append(selected_features)

            return population_value

        elif self.purpose == 'Parameter+Feature':

            population_value = []

            # 调参的种群数值
            parameter_chromosome_length = self.population_generate()[0]
            parameter_population_value = np.empty(shape=(self.population_size, len(parameter_chromosome_length)))
            # 记录参数值
            for i in range(self.population_size):  # 循环种群每个个体
                copied_population = population[i][:np.sum(parameter_chromosome_length)].copy()
                for j, num in enumerate(parameter_chromosome_length):  # 循环一个种群个体的每个参数染色体
                    new_chromosome = copied_population[:num]
                    strs = ''.join(str(int(k)) for k in new_chromosome)
                    int_value = int(strs, 2)
                    # 参数值
                    upper_bound = list(self.params_bound.values())[j][1]
                    lower_bound = list(self.params_bound.values())[j][0]
                    para_value = round(lower_bound + int_value * (upper_bound - lower_bound) / (pow(2, num) - 1))
                    parameter_population_value[i, j] = para_value
                    copied_population = np.delete(copied_population, range(num))

            # 过滤法
            if self.filter is not None:
                must_select_features = self.population_generate()[1]
                selected_ga_features = self.population_generate()[2]
                feature_population = self.population_generate()[3]

                for i in range(self.population_size):
                    selected_features = np.append(must_select_features,
                                                  selected_ga_features[feature_population[i, ::] == 1])
                    population_value.append(np.append(parameter_population_value[i, ::].astype(int), selected_features).astype(int))

                return population_value

            # 不过滤
            else:
                chromosome_length = self.population_generate()[2].shape[1]
                feature_population = self.population_generate()[2]
                for i in range(self.population_size):
                    selected_features = np.arange(chromosome_length)[feature_population[i, ::] == 1]

                    population_value.append(np.append(parameter_population_value[i, ::].astype(int), selected_features.astype(int)))

                return population_value

    # 计算适应度值函数
    def fitness_value(self, population_value):

        fitness_value = []

        # 针对KNN模型
        if self.model == 'KNN':  # GA用作调参
            if self.purpose == 'Parameter':  # GA用作调参
                for i in range(len(population_value)):
                    # 选择模型
                    model = KNeighborsRegressor(n_neighbors=int(population_value[i, 0]),
                                                leaf_size=int(population_value[i, 1]),
                                                n_jobs=5)
                    cv_scores = model_selection.cross_val_score(model,
                                                                self.x,
                                                                self.y,
                                                                cv=self.cv,
                                                                scoring='r2',
                                                                n_jobs=self.cv
                                                                )
                    # 适应值为不同参数组合下交叉验证值
                    fitness_value.append(cv_scores.mean())

            elif self.purpose == 'Feature':  # GA用作特征选择
                model = KNeighborsRegressor(**self.params, n_jobs=5)

                for i in range(self.population_size):

                    cv_scores = model_selection.cross_val_score(model,
                                                                self.x[:, population_value[i].astype(int)],
                                                                self.y,
                                                                cv=self.cv,
                                                                scoring='r2',
                                                                n_jobs=self.cv
                                                                )
                    fitness_value.append(cv_scores.mean())

            elif self.purpose == 'Parameter+Feature':
                for i in range(len(population_value)):
                    # 选择模型
                    model = KNeighborsRegressor(n_neighbors=int(population_value[i][0]),
                                                leaf_size=int(population_value[i][1]),
                                                n_jobs=5)

                    cv_scores = model_selection.cross_val_score(model,
                                                                self.x[:, population_value[i][len(self.params_bound):].astype(int)],
                                                                self.y,
                                                                cv=self.cv,
                                                                scoring='r2',
                                                                n_jobs=self.cv
                                                                )
                    # 适应值为不同参数组合下交叉验证值
                    fitness_value.append(cv_scores.mean())

        # 针对MLP模型
        elif self.model == 'MLP':
            if self.purpose == 'Parameter':  # GA用作调参
                for i in range(len(population_value)):
                    # 选择模型
                    model = MLPRegressor(hidden_layer_sizes=population_value[i][0], tol=1e-2)
                    cv_scores = model_selection.cross_val_score(model,
                                                                self.x,
                                                                self.y,
                                                                cv=self.cv,
                                                                scoring='r2',
                                                                n_jobs=self.cv)
                    # 适应值为不同参数组合下交叉验证值
                    fitness_value.append(cv_scores.mean())

            elif self.purpose == 'Feature':  # GA用作特征选择
                model = MLPRegressor(**self.params, tol=1e-2)
                for i in range(self.population_size):
                    cv_scores = model_selection.cross_val_score(model,
                                                                self.x[:, population_value[i]],
                                                                self.y,
                                                                cv=self.cv,
                                                                scoring='r2',
                                                                n_jobs=self.cv)
                    fitness_value.append(cv_scores.mean())

            elif self.purpose == 'Parameter+Feature':
                for i in range(self.population_size):
                    # 选择模型
                    model = MLPRegressor(hidden_layer_sizes=population_value[i][0], tol=1e-2)
                    cv_scores = model_selection.cross_val_score(model,
                                                                self.x[:, population_value[i][len(self.params_bound):].astype(int)],
                                                                self.y,
                                                                cv=self.cv,
                                                                scoring='r2',
                                                                n_jobs=self.cv
                                                                )
                    # 适应值为不同参数组合下交叉验证值
                    fitness_value.append(cv_scores.mean())

        return fitness_value

    # 轮盘赌选择操作
    def selection(self, population, fitness_value):

        # 计算总适应度值
        total = sum(fitness_value)

        # 适应值所占比例
        new_fitness = fitness_value / total

        # 适应度比例累加列表
        accumulated_fitness = []
        temp = 0
        for i in range(len(fitness_value)):
            temp += new_fitness[i]
            accumulated_fitness.append(temp)

        # 随机个random数 从种群中选取
        selected_population = np.empty(shape=(self.population_size, population.shape[1]))

        for i in range(self.population_size):
            select_criteria = random.random()
            for j in range(self.population_size):
                if accumulated_fitness[j] > select_criteria:
                    selected_population[i, ::] = population[j, ::]
                    break

        return selected_population.astype(int)

    # 交叉操作
    def cross_over(self, population):

        chromosome_length = population.shape[1]

        # 新建子代种群array
        offspring_population = np.empty(shape=(self.population_size, chromosome_length)).astype(int)

        for i in range(0, self.population_size, 2):  # 遍历种群

            # 交叉概率阈值
            cross_over_prob = random.random()

            father = population[random.randint(0, self.population_size) - 1, ::]  # 随机选择父代
            mother = population[random.randint(0, self.population_size) - 1, ::]  # 随机选择母代

            child1 = father
            child2 = mother

            # 如果小于交换概率 那么进行交换
            if cross_over_prob < self.pc:

                # 决定单点交叉还是两点交叉
                cross_over_type = random.random()

                if cross_over_type <= 1/3:  # 决定单点交叉还是多点交叉判断

                    cross_point = random.randint(0, chromosome_length - 1)  # 随机生成交换点位

                    child1[cross_point:] = mother[cross_point:]
                    child2[:cross_point] = father[:cross_point]

                    offspring_population[i, ::] = child1  # 子代一
                    offspring_population[i+1, ::] = child2  # 子代二

                elif 1/3 < cross_over_type <= 2/3:

                    cross_points = random.sample(range(0, chromosome_length - 1), 2)  # 交叉点位

                    child1[cross_points[0]:cross_points[1]] = mother[cross_points[0]:cross_points[1]]
                    child2[cross_points[0]:cross_points[1]] = father[cross_points[0]:cross_points[1]]

                    offspring_population[i, ::] = child1  # 子代一
                    offspring_population[i+1, ::] = child2  # 子代一

                else:

                    # 多点交叉
                    cross_points = random.sample(range(0, chromosome_length - 1),
                                                        random.randint(1, chromosome_length - 1))

                    for k in range(len(cross_points)):
                        child1[cross_points[k]] = mother[cross_points[k]]
                        child2[cross_points[k]] = father[cross_points[k]]

                    offspring_population[i, ::] = child1  # 子代一
                    offspring_population[i + 1, ::] = child2  # 子代一

            else:
                # 不交换
                # 多点对应交换
                father = population[random.randint(0, self.population_size) - 1, ::]  # 随机选择母代
                mother = population[random.randint(0, self.population_size) - 1, ::]  # 随机选择母代

                child1 = father
                child2 = mother

                offspring_population[i, ::] = child1  # 子代一
                offspring_population[i + 1, ::] = child2  # 子代一

        return offspring_population.astype(int)

    # 变异操作
    def mutation(self, population):

        chromosome_length = population.shape[1]

        for i in range(self.population_size):  # 遍历种群

            mutation_prob = random.random()  # 随机变异概率

            # 如果变异
            if mutation_prob < self.pm:

                mutation_points = random.sample(range(0, chromosome_length - 1), 2)  # 交叉串点位

                for j in range(mutation_points[0], mutation_points[1]+1):
                    population[i, j] = 1 - population[i, j]

        return population.astype(int)

    # 返回最好解
    def best(self, population_value, fitness_value):

        best_fitness = np.max(fitness_value)
        best_fitness_location = np.argmax(fitness_value)
        best_solution = population_value[best_fitness_location]

        return best_fitness, best_solution

    # 画图
    def plot(self, results):

        x = []
        y = []
        for i in range(self.iter_num):
            x.append(i + 1)
            y.append(results[i])
        plt.plot(x, y)

        plt.xlabel('Number of Iteration')
        plt.ylabel('Cross Validation Score (R squared)')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        if self.purpose == 'Parameter':
            plt.title('GA_' + self.model + ' for ' + 'Hyperparameter Tuning')
        else:
            plt.title('GA_' + self.model + ' for ' + 'Feature Selection')
        plt.show()

    # 主函数
    def run(self):

        # 记录结果
        results = []
        parameters = []
        best_fitness = 0.0
        best_parameters = []

        # 初始化种群

        # parameter
        if self.purpose == 'Parameter':
            population = self.population_generate()[1]

        # feature
        elif self.purpose == 'Feature':
            if self.filter is not None:
                population = self.population_generate()[2]
            else:
                population = self.population_generate()

        elif self.purpose == 'Parameter+Feature':
            if self.filter is not None:
                population = self.population_generate()[4]
            else:
                population = self.population_generate()[3]

        # 迭代参数寻优
        i = 0

        while i < self.iter_num:

            # 计算当前种群每个染色体的10进制取值
            population_value = self.decoding(population)

            # 计算适应函数
            fitness_value = self.fitness_value(population_value)

            # 寻找当前种群最好的参数值和最优适应度函数值
            current_fitness, current_parameters = self.best(population_value, fitness_value)

            # 与之前的最优适应度函数值比较，如果更优秀则替换最优适应度函数值和对应的参数
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_parameters = current_parameters

            print('iteration is :', i+1, ';最佳参数:', best_parameters, ';最佳适应值', best_fitness)
            results.append(best_fitness)
            parameters.append(best_parameters)

            # 种群更新

            # 选择
            after_selection_population = self.selection(population, fitness_value)

            # 交叉
            after_cross_over_population = self.cross_over(after_selection_population)

            # 变异
            after_mutation_population = self.mutation(after_cross_over_population)

            # 替换种群
            population = after_mutation_population

            i += 1

        global_optimal_results = max(results)
        global_optimal_solution = parameters[results.index(max(results))]

        print('全局最优结果为: ', global_optimal_results)
        print('全局最优参数为: ', global_optimal_solution)

        results.sort()
        self.plot(results)

        return global_optimal_solution

    # 将GA最优参数对模型test集进行训练
    def main(self, ga_results, params, x_train, y_train, x_test, y_test):
        global model
        # 将得到的最优参数再进行超参数的随机调参

        # 如果GA用于特征选择
        # 得到最优特征 再进行随机搜索 检索最优参数
        if self.purpose == 'Feature':

            # 如果是KNN模型
            if self.model == 'KNN':
                model = KNeighborsRegressor()
                model = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5,
                                           scoring='r2', n_iter=iter, n_jobs=-1)
                model.fit(x_train[:, ga_results], y_train)

                best_params = model.best_params_

                model = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'],
                                            leaf_size=best_params['leaf_size'], n_jobs=-1)

                model.fit(x_train[:, ga_results], y_train)

            # 如果是MLP模型
            elif self.model == 'MLP':

                model = KNeighborsRegressor()
                model = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5,
                                           scoring='r2', n_iter=iter, n_jobs=-1)

                model.fit(x_train[:, ga_results], y_train)

                best_params = model.best_params_

                model = MLPRegressor(hidden_layer_sizes=best_params['hidden_layer_sizes'])

                model.fit(x_train[:, ga_results], y_train)

            y_pred = model.predict(x_test[:, ga_results])
            y_pred = y_scale.inverse_transform(y_pred.reshape(-1, 1))
            y_test = y_scale.inverse_transform(y_test.reshape(-1, 1))
            r2 = r2_score(y_test, y_pred)
            print('R square: ', "%.3f" % r2)

        # 如果GA用于参数调参
        # 得到最优参数直接投入训练
        if self.purpose == 'Parameter':

            # 如果是KNN模型
            if self.model == 'KNN':

                model = KNeighborsRegressor(n_neighbors=ga_results[0],
                                            leaf_size=ga_results[1], n_jobs=-1)

                model.fit(x_train, y_train)

            # 如果是MLP模型
            if self.model == 'MLP':

                model = MLPRegressor(hidden_layer_sizes=ga_results[0])
                model.fit(x_train, y_train)

            y_pred = model.predict(x_test[:, ga_results])
            y_pred = y_scale.inverse_transform(y_pred.reshape(-1, 1))
            y_test = y_scale.inverse_transform(y_test.reshape(-1, 1))
            r2 = r2_score(y_test, y_pred)
            print('R square for ' + self.model, "%.3f" % r2)


if __name__ == "__main__":
    # Import data
    data = pd.read_csv('Final_data.csv')
    #
    housing_x = data.iloc[:, :-1]
    housing_y = data.iloc[:, -1].values.reshape(-1, 1)

    x_scale = StandardScaler().fit(housing_x.values)
    y_scale = StandardScaler().fit(housing_y)
    x_data = x_scale.transform(housing_x.values)
    y_data = y_scale.transform(housing_y)

    # Data split size
    size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data.flatten(), test_size=size, random_state=0)

    # Random search parameter
    iter = 20

    # RFECV parameter
    step = 3
    cv = 5

    # avm = Avm(housing_x, housing_y, size, step, cv)

    # params_bound={'n_neighbors': [3, 15], 'leaf_size': [10, 50]}

    # GA function
    ga = GA(x=x_train, y=y_train,
            model='KNN',
            purpose='Feature',
            params={'n_neighbors': 5, 'leaf_size': 30},
            population_size=20, iter_num=50, pc=0.8, pm=0.2,
            filter='MI')

    population = ga.population_generate()







