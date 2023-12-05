import pandas as pd
import numpy as np
import math
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
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from scipy.stats import uniform
import matplotlib.pyplot as plt
import shap
import random
import warnings
from time import time
from matplotlib.ticker import MaxNLocator
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

'''
Testing model include MLR, SVR, RF, Extra Trees, Adaboost, XGBoost, KNN, and MLP.
'''

# Machine Learning Model (MLR, SVR, RF, Extra Trees, Adaboost, XGBoost)
class AVM(object):
    # Parameter
    def __init__(self, x, y, test_size=0.2, cv=5, rs_iter=20):
        self.size = test_size
        self.cv = cv
        self.rs_iter = rs_iter
        self.x_scale = StandardScaler().fit(x.values)
        self.y_scale = StandardScaler().fit(y.values.reshape(-1, 1))
        self.x = self.x_scale.transform(x.values)
        self.y = self.y_scale.transform(y.values.reshape(-1, 1))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y.flatten(),
                                                                                test_size=self.size, random_state=0)

    # model performance evaluation
    def model_performance(self, y_test, y_pred):
        # Model performance evaluation

        # ML perspective
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mdape = np.median(np.abs((y_test-y_pred)/y_test))

        # Property valuation perspective
        ratio = y_pred / y_test
        ratio_median = np.median(ratio)
        ratio_mean = np.mean(ratio)
        weighted_ratio = np.sum(y_pred) / np.sum(y_test)
        cod = np.sum(abs(ratio - ratio_median)) / (len(ratio) * ratio_median) * 100

        # Model performance results
        performance_result = {'RMSE': "%.2f" % math.sqrt(mse),
                               '%RMSE': "%.2f" % (((math.sqrt(mse)) / abs(np.mean(y_test))) * 100) + '%',
                               'MAE': "%.2f" % mae,
                               '%MAE': "%.2f" % ((mae / abs(np.mean(y_test))) * 100) + '%',
                               'MAPE': "%.2f" % mape,
                               'MedAE': "%.2f" % medae,
                               'MdAPE': "%.2f" % mdape,
                               'R2': "%.2f" % r2,
                               'COD': "%.2f" % cod,
                               'Median Ratio': "%.2f" % ratio_median,
                               'Mean Ratio': "%.2f" % ratio_mean,
                               'Weighted Ratio': weighted_ratio}

        return performance_result

    # Select the best model set
    def best_model(self, optimal_features, model_results):
        benchmark_P0_F0 = {'R2': float(model_results['All_features_results']['R2']),
                           'COD': float(model_results['All_features_results']['COD'])
                           }
        performance_P0_F = {'R2': float(model_results['RS_before']['R2']),
                            'COD': float(model_results['RS_before']['COD'])
                            }
        performance_P_F = {'R2': float(model_results['RS_after']['R2']),
                           'COD': float(model_results['RS_after']['COD'])
                           }

        r2 = [benchmark_P0_F0['R2'], performance_P0_F['R2'], performance_P_F['R2']]
        cod = [benchmark_P0_F0['COD'], performance_P0_F['COD'], performance_P_F['COD']]

        # R2 is same
        if len(set(r2)) == 1:

            # See the value of COD
            # if COD is same, choose the second model set
            if len(set(cod)) == 1:
                features = optimal_features
                parameter = model_results['All_features_results']['Best parameters']
                performance = model_results['RS_before']

                return features, parameter, performance

            else:
                cod_model_index = cod.index(min(cod))
                if cod_model_index == 0:
                    features = [x for x in range(self.x.shape[1])]
                    parameter = model_results['All_features_results']['Best parameters']
                    performance = model_results['All_features_results']
                elif cod_model_index == 1:
                    features = optimal_features
                    parameter = model_results['All_features_results']['Best parameters']
                    performance = model_results['RS_before']
                else:
                    features = optimal_features
                    parameter = model_results['RS_after']['Best parameters']
                    performance = model_results['RS_after']

                return features, parameter, performance

        else:
            r2_model_index = r2.index(max(r2))
            if r2_model_index == 0:
                features = [x for x in range(self.x.shape[1])]
                parameter = model_results['All_features_results']['Best parameters']
                performance = model_results['All_features_results']
            elif r2_model_index == 1:
                features = optimal_features
                parameter = model_results['All_features_results']['Best parameters']
                performance = model_results['RS_before']
            else:
                features = optimal_features
                parameter = model_results['RS_after']['Best parameters']
                performance = model_results['RS_after']

            return features, parameter, performance

    # Multiple Linear Regression
    def MLR(self):

        # No need for parameter search

        # Model definition
        model = LinearRegression(n_jobs=-1)

        # Model training
        train_start = time()
        model.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = model.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        # Model performance results
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'

        # Select the optimal feature
        ref = RFECV(estimator=model, step=self.ref_step, scoring='r2', cv=self.cv, n_jobs=self.cv)

        rfecv_start = time()
        ref.fit(self.x_train, self.y_train)
        rfecv_finish = time()

        after_refcv_features = {'Features Num': ref.n_features_,
                                'Selected features': ref.ranking_,
                                'RFECV time': "%.2f" % (rfecv_finish - rfecv_start) + 's'}

        # Use the features after REFCV to train the model
        optimal_features = [x for x, y in enumerate(ref.ranking_) if y == 1]
        new_train_x = self.x_train[:, optimal_features]
        new_train_y = self.y_train

        # New train time
        new_train_start = time()
        model.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = model.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        RFECV_features_result = self.model_performance(y_test, new_y_pred)
        RFECV_features_result['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'

        model_results = {'All_features_results': all_features_result,
                         'RFECV_RS_before': after_refcv_features,
                         'RFECV_RS_after': RFECV_features_result}

        # REFCV results
        # plt.figure()
        # plt.xlabel('Number of features selected')
        # plt.ylabel('Mean cross validation score (R squared)')
        # plt.ylim(0, 1)
        # plt.plot([1] + list(reversed(range(self.x.shape[1], 1, -self.ref_step))), ref.cv_results_['mean_test_score'])
        # plt.show()
        # file = open('..\\paper codes\\result\\MLR.txt', 'w')
        # for k, v in model_results.items():
        #     file.write(str(k) + ' ' + str(v) + '\n')
        # file.close()

        return model_results

    # Linear SVR
    def Linear_SVR(self,lower_bound, upper_bound, rfecv_step):
        # Random search parameters
        parameters = {'epsilon': uniform(loc=lower_bound[0], scale=upper_bound[0]), 'C': uniform(loc=lower_bound[1], scale=upper_bound[1])}

        # model definition
        model = LinearSVR(random_state=0,  max_iter=10000, tol=0.01)

        para_search_start = time()
        best_para_model0 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv, scoring='r2',
                                              n_iter=self.rs_iter, n_jobs=self.cv)
        best_para_model0.fit(self.x_train, self.y_train)
        para_search_finish = time()

        optimal_parameter0 = best_para_model0.best_params_

        train_start = time()
        best_para_model0 = LinearSVR(random_state=0, epsilon=optimal_parameter0['epsilon'], C=optimal_parameter0['C'],
                                     max_iter=10000, tol=0.01)
        best_para_model0.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = best_para_model0.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        """
        P0 and F0
        """
        # Model performance evaluation
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'
        all_features_result['Best parameters']= optimal_parameter0
        all_features_result['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        # Select the optimal features
        ref = RFECV(estimator=best_para_model0, step=rfecv_step, scoring='r2', cv=self.cv, n_jobs=self.cv)

        rfecv_start = time()
        ref.fit(self.x_train, self.y_train)
        rfecv_finish = time()

        # Use the features after REFCV to train the model
        optimal_features = [x for x, y in enumerate(ref.ranking_) if y == 1]

        # optimal parameters after REFCV
        new_train_x = self.x_train[:, optimal_features]
        new_train_y = self.y_train

        '''
        P0 and F
        '''
        best_para_model0.fit(new_train_x, new_train_y)
        current_y_pred = best_para_model0.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        current_y_pred = self.y_scale.inverse_transform(current_y_pred.reshape(-1, 1))

        # Model performance results
        after_refcv_results = self.model_performance(y_test, current_y_pred)
        after_refcv_features = {'Features Num': ref.n_features_,
                                'Selected features': ref.ranking_,
                                'RFECV time': "%.2f" % (rfecv_finish - rfecv_start) + 's'}

        RFECV_RS_before = {**after_refcv_features, **after_refcv_results}

        '''
        P and F
        '''
        # Random search or grid search on the optimal features
        best_para_model1 = RandomizedSearchCV(estimator=best_para_model0, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)

        para_search_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        para_search_finish = time()

        optimal_parameter1 = best_para_model1.best_params_
        best_para_model1 = LinearSVR(epsilon=optimal_parameter1['epsilon'], C=optimal_parameter1['C'], random_state=0,
                                     max_iter=10000, tol=0.01)

        new_train_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = best_para_model1.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        RFECV_RS_after = self.model_performance(y_test, new_y_pred)
        RFECV_RS_after['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'
        RFECV_RS_after['Best parameters'] = optimal_parameter1
        RFECV_RS_after['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        Summary
        '''
        model_results = {'All_features_results': all_features_result,
                         'RS_before': RFECV_RS_before,
                         'RS_after': RFECV_RS_after}

        # Store as txt file
        # file = open('..\\paper codes\\result\\Linear_SVR.txt', 'w')
        # for k, v in model_results.items():
        #     file.write(str(k) + ' ' + str(v) + '\n')
        # file.close()
        model_results = self.best_model(optimal_features, model_results)

        return model_results

    # Random forest
    def RF(self, lower_bound, upper_bound, rfecv_step):
        # Random search or Grid search parameters
        parameters = {'n_estimators': [x for x in range(lower_bound[0], upper_bound[0]+1)], 'max_depth': [x for x in range(lower_bound[1], upper_bound[1]+1)]}

        # model definition
        model = RandomForestRegressor(random_state=0, n_jobs=-1)

        '''
        P0 and F0
        '''

        # Random search
        best_para_model0 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)

        # Searching
        para_search_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        para_search_finish = time()

        optimal_parameter0 = best_para_model0.best_params_

        # ML model definition
        best_para_model0 = RandomForestRegressor(random_state=0, n_estimators=optimal_parameter0['n_estimators'],
                                                 max_depth=optimal_parameter0['max_depth'], n_jobs=-1)
        # model training
        train_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = best_para_model0.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        # Model performance results
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'
        all_features_result['Best parameters'] = optimal_parameter0
        all_features_result['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        P0 and F
        '''
        # Select the optimal features
        ref = RFECV(estimator=best_para_model0, step=rfecv_step, scoring='r2', cv=self.cv, n_jobs=self.cv)

        rfecv_start = time()
        ref.fit(self.x_train, self.y_train)
        rfecv_finish = time()

        # Use the features after REFCV to train the model
        optimal_features = [x for x, y in enumerate(ref.ranking_) if y == 1]
        # optimal parameters after REFCV
        new_train_x = self.x_train[:, optimal_features]
        new_train_y = self.y_train

        best_para_model0.fit(new_train_x, new_train_y)
        current_y_pred = best_para_model0.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        current_y_pred = self.y_scale.inverse_transform(current_y_pred.reshape(-1, 1))

        # Model performance results
        after_refcv_results = self.model_performance(y_test, current_y_pred)
        after_refcv_features = {'Features Num': ref.n_features_,
                                'Selected features': ref.ranking_,
                                'RFECV time': "%.2f" % (rfecv_finish - rfecv_start) + 's'}

        RFECV_RS_before = {**after_refcv_features, **after_refcv_results}

        '''
        P and F
        '''
        # Random search or grid search on the optimal features
        best_para_model1 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        para_search_finish = time()

        optimal_parameter1 = best_para_model1.best_params_

        best_para_model1 = RandomForestRegressor(random_state=0, n_estimators=optimal_parameter1['n_estimators'],
                                                 max_depth=optimal_parameter1['max_depth'], n_jobs=-1)
        new_train_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = best_para_model1.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        RFECV_RS_after = self.model_performance(y_test, new_y_pred)
        RFECV_RS_after['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'
        RFECV_RS_after['Best parameters'] = optimal_parameter1
        RFECV_RS_after['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        Summary
        '''
        model_results = {'All_features_results': all_features_result,
                         'RS_before': RFECV_RS_before,
                         'RS_after': RFECV_RS_after}

        # Store as txt file
        # file = open('..\\paper codes\\result\\RF.txt', 'w')
        # for k, v in model_results.items():
        #      file.write(str(k) + ' ' + str(v) + '\n')
        #  file.close()

        model_results = self.best_model(optimal_features, model_results)

        return model_results

    # Extra tree
    def Extra_Tree(self, lower_bound, upper_bound, rfecv_step):
        # Random search or Grid search parameters
        parameters = {'n_estimators': [x for x in range(lower_bound[0], upper_bound[0]+1)], 'max_depth': [x for x in range(lower_bound[1], upper_bound[1]+1)]}

        # model definition
        model = ExtraTreesRegressor(random_state=0, n_jobs=-1)

        best_para_model0 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        para_search_finish = time()

        optimal_parameter0 = best_para_model0.best_params_


        best_para_model0 = ExtraTreesRegressor(random_state=0, n_estimators=optimal_parameter0['n_estimators'],
                                               max_depth=optimal_parameter0['max_depth'], n_jobs=-1)
        train_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = best_para_model0.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        # Model performance results
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'
        all_features_result['Best parameters'] = optimal_parameter0
        all_features_result['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        # Select the optimal features
        ref = RFECV(estimator=best_para_model0, step=rfecv_step, scoring='r2', cv=self.cv, n_jobs=self.cv)
        rfecv_start = time()
        ref.fit(self.x_train, self.y_train)
        rfecv_finish = time()

        # Use the features after REFCV to train the model
        optimal_features = [x for x, y in enumerate(ref.ranking_) if y == 1]
        # optimal parameters after REFCV
        new_train_x = self.x_train[:, optimal_features]
        new_train_y = self.y_train

        '''
        P0 and F
        '''
        best_para_model0.fit(new_train_x, new_train_y)
        current_y_pred = best_para_model0.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        current_y_pred = self.y_scale.inverse_transform(current_y_pred.reshape(-1, 1))

        # Model performance results
        after_refcv_results = self.model_performance(y_test, current_y_pred)
        after_refcv_features = {'Features Num': ref.n_features_,
                                'Selected features': ref.ranking_,
                                'RFECV time': "%.2f" % (rfecv_finish - rfecv_start) + 's'}

        RFECV_RS_before = {**after_refcv_features, **after_refcv_results}

        '''
        P and F
        '''
        # Random search or grid search on the optimal features
        best_para_model1 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        para_search_finish = time()

        optimal_parameter1 = best_para_model1.best_params_

        best_para_model1 = ExtraTreesRegressor(random_state=0, n_estimators=optimal_parameter1['n_estimators'],
                                               max_depth=optimal_parameter1['max_depth'], n_jobs=-1)
        new_train_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = best_para_model1.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        RFECV_RS_after = self.model_performance(y_test, new_y_pred)
        RFECV_RS_after['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'
        RFECV_RS_after['Best parameters'] = optimal_parameter1
        RFECV_RS_after['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        Summary
        '''
        model_results = {'All_features_results': all_features_result,
                         'RS_before': RFECV_RS_before,
                         'RS_after': RFECV_RS_after}


        # Store as txt file
        # file = open('..\\paper codes\\result\\Extra_Tree.txt', 'w')
        # for k, v in model_results.items():
        #     file.write(str(k) + ' ' + str(v) + '\n')
        # file.close()

        model_results = self.best_model(optimal_features, model_results)

        return model_results

    # AdaBoost
    def AdaBoost(self, lower_bound, upper_bound, rfecv_step):
        # Random search or Grid search parameters
        parameters = {'n_estimators': [x for x in range(lower_bound[0], upper_bound[0]+1)], 'learning_rate': uniform(loc=lower_bound[1], scale=upper_bound[1]+1)}

        # model definition
        model = AdaBoostRegressor(random_state=0)

        # First round searching
        best_para_model0 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        para_search_finish = time()

        optimal_parameter0 = best_para_model0.best_params_
        best_para_model0 = AdaBoostRegressor(random_state=0, n_estimators=optimal_parameter0['n_estimators'],
                                             learning_rate=optimal_parameter0['learning_rate'])
        train_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = best_para_model0.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        # Model performance results
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'
        all_features_result['Best parameters'] = optimal_parameter0
        all_features_result['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        # Select the optimal features
        ref = RFECV(estimator=best_para_model0, step=rfecv_step, scoring='r2', cv=self.cv, n_jobs=self.cv)
        rfecv_start = time()
        ref.fit(self.x_train, self.y_train)
        rfecv_finish = time()

        # Use the features after REFCV to train the model
        optimal_features = [x for x, y in enumerate(ref.ranking_) if y == 1]
        # optimal parameters after REFCV
        new_train_x = self.x_train[:, optimal_features]
        new_train_y = self.y_train

        '''
        P0 and F
        '''
        best_para_model0.fit(new_train_x, new_train_y)
        current_y_pred = best_para_model0.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        current_y_pred = self.y_scale.inverse_transform(current_y_pred.reshape(-1, 1))

        # Model performance results
        after_refcv_results = self.model_performance(y_test, current_y_pred)
        after_refcv_features = {'Features Num': ref.n_features_,
                                'Selected features': ref.ranking_,
                                'RFECV time': "%.2f" % (rfecv_finish - rfecv_start) + 's'}

        RFECV_RS_before = {**after_refcv_features, **after_refcv_results}

        '''
        P and F
        '''
        # Random search or grid search on the optimal features
        best_para_model1 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        para_search_finish = time()

        optimal_parameter1 = best_para_model1.best_params_
        best_para_model1 = AdaBoostRegressor(random_state=0, n_estimators=optimal_parameter1['n_estimators'],
                                             learning_rate=optimal_parameter1['learning_rate'])
        new_train_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = best_para_model1.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        RFECV_RS_after = self.model_performance(y_test, new_y_pred)
        RFECV_RS_after['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'
        RFECV_RS_after['Best parameters'] = optimal_parameter1
        RFECV_RS_after['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        Summary
        '''
        model_results = {'All_features_results': all_features_result,
                         'RS_before': RFECV_RS_before,
                         'RS_after': RFECV_RS_after}

        # Store as txt file
        # file = open('..\\paper codes\\result\\Adaboost.txt', 'w')
        # for k, v in model_results.items():
        #     file.write(str(k) + ' ' + str(v) + '\n')
        # file.close()

        model_results = self.best_model(optimal_features, model_results)

        return model_results

    # XGBoost
    def XGBoost(self, lower_bound, upper_bound, rfecv_step):
        # Random search or Grid search parameters
        parameters = {'n_estimators': [x for x in range(lower_bound[0], upper_bound[0]+1)], 'max_depth': [x for x in range(lower_bound[1], upper_bound[1]+1)]}

        # model definition
        model = XGBRegressor(random_state=0, n_jobs=-1)

        best_para_model0 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        para_search_finish = time()

        optimal_parameter0 = best_para_model0.best_params_
        best_para_model0 = XGBRegressor(random_state=0, n_estimators=optimal_parameter0['n_estimators'],
                                        max_depth=optimal_parameter0['max_depth'], n_jobs=-1)
        train_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = best_para_model0.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        # Model performance results
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'
        all_features_result['Best parameters'] = optimal_parameter0
        all_features_result['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        # Select the optimal features
        ref = RFECV(estimator=best_para_model0, step=rfecv_step, scoring='r2', cv=self.cv, n_jobs=self.cv)
        rfecv_start = time()
        ref.fit(self.x_train, self.y_train)
        rfecv_finish = time()

        # Use the features after REFCV to train the model
        optimal_features = [x for x, y in enumerate(ref.ranking_) if y == 1]
        # optimal parameters after REFCV
        new_train_x = self.x_train[:, optimal_features]
        new_train_y = self.y_train

        '''
        P0 and F
        '''
        best_para_model0.fit(new_train_x, new_train_y)
        current_y_pred = best_para_model0.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        current_y_pred = self.y_scale.inverse_transform(current_y_pred.reshape(-1, 1))

        # Model performance results
        after_refcv_results = self.model_performance(y_test, current_y_pred)
        after_refcv_features = {'Features Num': ref.n_features_,
                                'Selected features': ref.ranking_,
                                'RFECV time': "%.2f" % (rfecv_finish - rfecv_start) + 's'}

        RFECV_RS_before = {**after_refcv_features, **after_refcv_results}

        '''
        P and F
        '''
        # Random search or grid search on the optimal features
        best_para_model1 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        para_search_finish = time()

        optimal_parameter1 = best_para_model1.best_params_
        best_para_model1 = XGBRegressor(random_state=0, n_estimators=optimal_parameter1['n_estimators'],
                                        max_depth=optimal_parameter1['max_depth'], n_jobs=-1)
        new_train_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = best_para_model1.predict(self.x_test[:, optimal_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        RFECV_RS_after = self.model_performance(y_test, new_y_pred)
        RFECV_RS_after['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'
        RFECV_RS_after['Best parameters'] = optimal_parameter1
        RFECV_RS_after['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        Summary
        '''
        model_results = {'All_features_results': all_features_result,
                         'RS_before': RFECV_RS_before,
                         'RS_after': RFECV_RS_after}

        # Store as txt file
        # file = open('..\\paper codes\\result\\XGBoost.txt', 'w')
        # for k, v in model_results.items():
        #     file.write(str(k) + ' ' + str(v) + '\n')
        # file.close()
        model_results = self.best_model(optimal_features, model_results)

        return model_results

    # KNN
    def KNN(self, lower_bound, upper_bound, population_size, iter_num, pc, pm, filter=None):

        # Random search or Grid search parameters
        parameters = {'n_neighbors': [x for x in range(lower_bound[0], upper_bound[0]+1)], 'leaf_size': [x for x in range(lower_bound[1], upper_bound[1]+1)]}

        # model definition
        model = KNeighborsRegressor(n_jobs=-1)

        best_para_model0 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        para_search_finish = time()

        optimal_parameter0 = best_para_model0.best_params_
        best_para_model0 = KNeighborsRegressor(n_neighbors=optimal_parameter0['n_neighbors'],
                                               leaf_size=optimal_parameter0['leaf_size'],
                                               n_jobs=-1)
        train_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = best_para_model0.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        # Model performance results
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'
        all_features_result['Best parameters'] = optimal_parameter0
        all_features_result['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        # GA searching
        print('GA Searching Start!')
        ga_start = time()
        ga_features = GA(x=self.x_train, y=self.y_train,
                         model='KNN',
                         purpose='Feature',
                         params=optimal_parameter0,
                         cv=self.cv,
                         population_size=population_size, iter_num=iter_num, pc=pc, pm=pm,
                         filter=filter).run()
        ga_finish = time()

        # optimal parameters after GA
        new_train_x = self.x_train[:, ga_features]
        new_train_y = self.y_train

        '''
        P0 and F
        '''
        best_para_model0.fit(new_train_x, new_train_y)
        current_y_pred = best_para_model0.predict(self.x_test[:, ga_features])

        # true values and prediction values
        current_y_pred = self.y_scale.inverse_transform(current_y_pred.reshape(-1, 1))

        # Model performance results
        after_ga_results = self.model_performance(y_test, current_y_pred)
        after_ga_features = {'Features Num': len(ga_features),
                             'Selected features': ga_features,
                             'GA searching time': "%.2f" % (ga_finish - ga_start) + 's'}

        GA_RS_before = {**after_ga_features, **after_ga_results}

        '''
        P and F
        '''
        # Random search or grid search on the optimal features
        best_para_model1 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        para_search_finish = time()

        optimal_parameter1 = best_para_model1.best_params_
        best_para_model1 = KNeighborsRegressor(n_neighbors=optimal_parameter1['n_neighbors'],
                                               leaf_size=optimal_parameter1['leaf_size'], n_jobs=-1)
        new_train_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = best_para_model1.predict(self.x_test[:, ga_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        GA_RS_after = self.model_performance(y_test, new_y_pred)
        GA_RS_after['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'
        GA_RS_after['Best parameters'] = optimal_parameter1
        GA_RS_after['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        Summary
        '''
        model_results = {'All_features_results': all_features_result,
                         'RS_before': GA_RS_before,
                         'RS_after': GA_RS_after}

        # Store as txt file
        # file = open('..\\paper codes\\result\\KNN.txt', 'w')
        # for k, v in model_results.items():
        #     file.write(str(k) + ' ' + str(v) + '\n')
        # file.close()

        model_results = self.best_model(ga_features, model_results)

        return model_results

    # MLP
    def MLP(self, lower_bound, upper_bound, population_size, iter_num, pc, pm, filter=None):
        # Random search or Grid search parameters
        parameters = {'hidden_layer_sizes': [x for x in range(lower_bound[0],  upper_bound[0]+1)]}

        # model definition
        model = MLPRegressor(tol=0.01, max_iter=5000)

        best_para_model0 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        para_search_finish = time()

        optimal_parameter0 = best_para_model0.best_params_
        best_para_model0 = MLPRegressor(hidden_layer_sizes=optimal_parameter0['hidden_layer_sizes'])

        train_start = time()
        best_para_model0.fit(self.x_train, self.y_train)
        train_finish = time()

        # Model prediction
        y_pred = best_para_model0.predict(self.x_test)

        # true values and prediction values
        y_pred = self.y_scale.inverse_transform(y_pred.reshape(-1, 1))
        y_test = self.y_scale.inverse_transform(self.y_test.reshape(-1, 1))

        # Model performance results
        all_features_result = self.model_performance(y_test, y_pred)
        all_features_result['Train time'] = "%.2f" % (train_finish - train_start) + 's'
        all_features_result['Best parameters'] = optimal_parameter0
        all_features_result['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        # GA searching
        print('GA Searching Start!')
        ga_start = time()
        ga_features = GA(x=self.x_train, y=self.y_train,
                         model='MLP',
                         purpose='Feature',
                         params=optimal_parameter0,
                         population_size=population_size, iter_num=iter_num, pc=pc, pm=pm,
                         filter=filter).run()
        ga_finish = time()

        # optimal parameters after REFCV
        new_train_x = self.x_train[:, ga_features]
        new_train_y = self.y_train

        '''
        P0 and F
        '''
        best_para_model0.fit(new_train_x, new_train_y)
        current_y_pred = best_para_model0.predict(self.x_test[:, ga_features])

        # true values and prediction values
        current_y_pred = self.y_scale.inverse_transform(current_y_pred.reshape(-1, 1))

        # Model performance results
        after_ga_results = self.model_performance(y_test, current_y_pred)
        after_ga_features = {'Features Num': len(ga_features),
                             'Selected features': ga_features,
                             'GA searching time': "%.2f" % (ga_finish - ga_start) + 's'}

        GA_RS_before = {**after_ga_features, **after_ga_results}

        '''
        P and F
        '''
        # Random search or grid search on the optimal features
        best_para_model1 = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=self.cv,
                                              scoring='r2', n_iter=self.rs_iter, n_jobs=self.cv)
        para_search_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        para_search_finish = time()

        optimal_parameter1 = best_para_model1.best_params_
        best_para_model1 = MLPRegressor(hidden_layer_sizes=optimal_parameter1['hidden_layer_sizes'],
                                        tol=0.01, max_iter=5000)

        new_train_start = time()
        best_para_model1.fit(new_train_x, new_train_y)
        new_train_finish = time()

        # Model prediction
        new_y_pred = best_para_model1.predict(self.x_test[:, ga_features])

        # true values and prediction values
        new_y_pred = self.y_scale.inverse_transform(new_y_pred.reshape(-1, 1))

        # Model performance results
        GA_RS_after = self.model_performance(y_test, new_y_pred)
        GA_RS_after['Train time'] = "%.2f" % (new_train_finish - new_train_start) + 's'
        GA_RS_after['Best parameters'] = optimal_parameter1
        GA_RS_after['Parameter search time'] = "%.2f" % (para_search_finish - para_search_start) + 's'

        '''
        Summary
        '''
        model_results = {'All_features_results': all_features_result,
                         'RS_before': GA_RS_before,
                         'RS_after': GA_RS_after}

        # Store as txt file
        # file = open('..\\paper codes\\result\\MLP.txt', 'w')
        # for k, v in model_results.items():
        #     file.write(str(k) + ' ' + str(v) + '\n')
        # file.close()

        model_results = self.best_model(ga_features, model_results)

        return model_results


# GA for feature selection and hyperparameter tuning
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

        # LinearSVR
        # RF
        # ExtraTree
        # AdaBoost
        #XGBoost

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
                    offspring_population[i+1, ::] = child2  # 子代一

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

        # results.sort()
        # self.plot(results)

        return global_optimal_solution


# Feature importance
class Explainability(object):
    # 定义模型
    def __init__(self, x, y, model, test_size=0.2, random_state=0):
        self.x_scaler = StandardScaler().fit(x.values)
        self.y_scaler = StandardScaler().fit(y.values.reshape(-1, 1))
        self.x = self.x_scaler.transform(x.values)
        self.y = self.y_scaler.transform(y.values.reshape(-1, 1))

        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,
                                                                                self.y.flatten(),
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)
        self.features = x.columns.values.tolist()

    # Global importance shap
    def shap_global_importance(self):
        # define the model
        model = self.model.fit(self.x_train, self.y_train)

        # define the shap explainer
        try:
            explainer = shap.Explainer(model, feature_names=self.features)
            shap_values = explainer(self.x_train)

        except TypeError:
            explainer = shap.KernelExplainer(model.predict, self.x_train)
            shap_values = explainer.shap_values(self.x_train)
        # 多个样本SHA集合

        shap.plots.bar(shap_values, max_display=len(self.features), show=False)

        plt.savefig('..\\paper codes\\templates\\results\\shap_global.png')

        shap_values = pd.DataFrame(shap_values.values)
        shap_values.columns = self.features
        shap_values.to_csv('..\\paper codes\\templates\\results\\shap_global.csv', index=False)

        # save the results
        return shap_values

    # Global importance model
    def model_global_importance(self):
        # define the model
        model = self.model.fit(self.x_train, self.y_train)

        # return the feature importance or coef_
        try:
            feature_importance = model.feature_importances_

        except AttributeError:
            feature_importance = model.coef_

        feature_importance = (feature_importance - np.min(feature_importance)) \
                             / (np.max(feature_importance) - np.min(feature_importance)) * 100

        feature_importance = pd.DataFrame(feature_importance).T

        feature_importance.columns = self.features

        feature_importance = feature_importance.T.sort_values(by=0, axis=0, ascending=False).copy()
        feature_importance.columns = ['Relative Importance']

        feature_importance.to_csv('..\\paper codes\\templates\\results\\model_global.csv')

        feature_importance.plot.barh(figsize=[14, 10])
        plt.xlabel('Relative Importance (%)')
        plt.ylabel('Features')
        plt.savefig('..\\paper codes\\templates\\results\\model_global.png', dpi=300)

        return feature_importance

    # Local importance SHAP
    def shap_local_importance(self, train_test, index):
        # define the model
        model = self.model.fit(self.x_train, self.y_train)

        # Train Set
        if train_test == 'Train Set':
            # define the shap explainer
            try:
                explainer = shap.Explainer(model, feature_names=self.features)
                values = explainer(self.x_train[index, ::])

                shap.initjs()
                p = shap.force_plot(self.y_scaler.inverse_transform(values.base_values.reshape(1, -1))[0, 0],
                                    values.values,
                                    features=self.x_scaler.inverse_transform(self.x_train[index, ::].reshape(1, -1)),
                                    feature_names=self.features,
                                    figsize=[20, 15],
                                    show=False)
                shap.save_html('..\\paper codes\\templates\\results\\local_train_shap.html', p)

                values = pd.DataFrame(values.values, columns=self.features)
                values.to_csv('..\\paper codes\\templates\\results\\local_train_shap.csv', index=False)

            # Knernal Explainer (for linear SVR)
            except TypeError:
                med = np.median(self.x_train, axis=0).reshape((1, self.x_train.shape[1]))
                explainer = shap.KernelExplainer(model.predict, med)
                values = explainer.shap_values(self.x_train[index, ::])

                shap.initjs()
                predicted_value = model.predict(self.x_train[index, ::].reshape(1, -1))

                raw_value = self.y_scaler.inverse_transform(predicted_value.reshape(1, -1))

                p = shap.force_plot(raw_value[0, 0],
                                    values,
                                    features=self.x_scaler.inverse_transform(self.x_train[index, ::].reshape(1, -1)),
                                    feature_names=self.features,
                                    figsize=[20, 15],
                                    show=False)

                shap.save_html('..\\paper codes\\templates\\results\\local_train_shap.html', p)

                values = pd.DataFrame(values, columns=self.features)
                values.to_csv('..\\paper codes\\templates\\results\\local_train_shap.csv', index=False)

        # Test Data Set
        else:
            try:
                explainer = shap.Explainer(model, feature_names=self.features)
                values = explainer(self.x_test[index, ::])
                shap.initjs()
                p = shap.force_plot(self.y_scaler.inverse_transform(values.base_values.reshape(1, -1))[0, 0],
                                    values.values,
                                    features=self.x_scaler.inverse_transform(self.x_test[index, ::].reshape(1, -1)),
                                    feature_names=self.features,
                                    figsize=[20, 15],
                                    show=False)
                shap.save_html('..\\paper codes\\templates\\results\\local_test_shap.html', p)

                values = pd.DataFrame(values.values, columns=self.features)
                values.to_csv('..\\paper codes\\templates\\results\\local_test_shap.csv', index=False)

            except TypeError:
                med = np.median(self.x_train, axis=0).reshape((1, self.x_train.shape[1]))
                explainer = shap.KernelExplainer(model.predict, med)
                values = explainer.shap_values(self.x_test[index, ::])

                shap.initjs()
                predicted_value = model.predict(self.x_test[index, ::].reshape(1, -1))

                raw_value = self.y_scaler.inverse_transform(predicted_value.reshape(1, -1))

                p = shap.force_plot(raw_value[0, 0],
                                    values,
                                    features=self.x_scaler.inverse_transform(self.x_test[index, ::].reshape(1, -1)),
                                    feature_names=self.features,
                                    figsize=[20, 15],
                                     show=False)
                shap.save_html('..\\paper codes\\templates\\results\\local_test_shap.html', p)

                values = pd.DataFrame(values, columns=self.features)
                values.to_csv('..\\paper codes\\templates\\results\\local_test_shap.csv', index=False)

        return values

    # Local importance LIME
    def lime_local_importance(self, train_test, index):

        # define the shap explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(self.x_train, mode='regression', feature_names=self.features)

        # shap results
        if train_test == 'Train Set':
            values = explainer.explain_instance(self.x_train[index],
                                                self.model.predict,
                                                num_features=self.x_train.shape[1])

        else:
            values = explainer.explain_instance(self.x_test[index],
                                                self.model.predict,
                                                num_features=self.x_train.shape[1])

        values.save_to_file('..\\paper codes\\templates\\results\\local_lime.html')


if __name__ == "__main__":
    data = pd.read_csv('Final_data.csv')
    housing_x = data.iloc[:, :-1]
    housing_y = data.iloc[:, -1]
    avm = AVM(housing_x, housing_y)





