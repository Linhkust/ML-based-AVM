



<h2 align="center">Machine-learning-based Automated Residential Property Valuation 
</h2>




## Abstract

Property valuation is an essential business in the real estate market. Traditional models have limited capability due to subjectivity and strict assumptions of normality, homoscedasticity, independence and non-multicollinearity. This paper has established a framework for systematically developing an automated property valuation (MLAPV) system that is quick, accurate, cost-effective, unbiased and consistent, integrating property valuation modeling principles, database creation, machine learning, systems development, and application. Novel approaches are provided on (1) collecting and integrating property, spatial and temporal data, (2) sequential feature selection and hyperparameter tuning by combining recursive feature elimination, cross-validation, genetic algorithm, filter, wrapper and random search, and (3) explainability of machine-learning results. A web-enabled MLAPV prototype is developed to demonstrate the framework’s applicability and a case study of the Hong Kong residential market conducted to prove the prototype’s usefulness. 



## Research Methodology

![Methodology](https://github.com/Linhkust/ML-based-AVM/blob/main/paper%20images/research%20framework.png)



## How to use the codes

### 1. Model selection

This paper mainly use scikit-learn library for ML modelling. Users can direct import the package in the Python codes:

```python
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost.sklearn import XGBRegressor  
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
```

Based on the imported ML models, this paper has defiend a Python class called `AVM` to initiate model training. The architecture of the class is expressed as below:

```py
class AVM(object):
    def MLR(self)
    def LinearSVR(self, lower_bound, upper_bound, rfecv_step)
    def RF(self, lower_bound, upper_bound, rfecv_step)
    def Extra_Tree(self, lower_bound, upper_bound, rfecv_step)
    def AdaBoost(self, lower_bound, upper_bound, rfecv_step)
    def XGBoost(self, lower_bound, upper_bound, rfecv_step)
    def KNN(self, lower_bound, upper_bound, population_size, iter_num, pc, pm, filter=None)
    def MLP(self, lower_bound, upper_bound, population_size, iter_num, pc, pm, filter=None)
```



### 2. Feature selection and Hyperparameter Tuning

We formulated a novel sequential optimization strategy for feature selection and hyperparameter tuning, integrating genetic algorithm, filter, wrapper, and random search. The followchart is illustrated as follows:

![FS and HPO](https://github.com/Linhkust/ML-based-AVM/blob/main/paper%20images/Sequential%20optimization%20strategy%20for%20feature%20selection%20and%20hyperparameter%20tuning.png)  

#### For ML models except KNN and MLP

Uses need to define the range of ML models' hyperparameter first, e.g., `n_estimators` and `max_depth` in `RandomForestRegressor` as follows:

```py
lower_bound=[50, 10], upper_bound=[150, 100]
```

This means that the range for `n_estimators` is 50-150 and `max_depth` is 10-100. Users then determine the parameter `rfecv_step` of `RFECV` for feature selection based on the hyperparameters obtained by first-round random search. A second round random search with the optimal features is implemented to find the optimal hyperparameters.  

#### For KNN and MLP

We define a Python class called `GA` in `GA_filter.py` for feature selection and hyperparameter tuning for KNN and MLP because these two ML models cannot use `RFECV` function. The architecture of the class is expressed as below:

```python
class GA(object):
    def population_generate(self)
    def decoding(self, population)
    def fitness_value(self, population_value)
    def selection(self, population, fitness_value)
    def cross_over(self, population)
    def mutation(self, population)
    def best(self, population_value, fitness_value)
    def plot(self, results)
    def run(self)  # main function
```

Based on the optimal hyperparameters of first round random search, optimal features of KNN and MLP are obtained using filter method first and GA search then. Filter method uses `mutual information` as criteria to obtain:

- `Remove set`: features with MI less than the 25th percentile (Q1)
- `GA select set`:  features with MI between Q1 and Q3
- `Select set`: features with MI larger than the 75th percentile (Q3)

Then, we initiate GA search using the parameters:

- `population_size`: size the population for GA search
- `iter_num`: iteration times
- `pc`: cross over rate
- `pm`: mutation rate  



### 3. Model Interpretability Analysis

We define a Python class called `Explainability` and the architecture is expressed as:

```py
class Explainability(object):
    # Global feature importance analysis
    def shap_global_importance(self)
    def model_global_importance(self)
    
    # Local feature importance analysis
    def shap_local_importance(self, train_test, index)
    def lime_local_importance(self, train_test, index)
```



### 4. Web Application Development

The architecture of the web application is as follows. The frontend is developed with HTML, CSS, and JavaScript to provide user interfaces. The backend is developed using Flask and Python. The Flask is used as a web server to receive HTTP requests from the frontend and return HTTP responses from the backend. The various functions of the system are supported by Python scripts. The database is created with MySQL. Google Map API can be incorporated to show the geographic locations and the street views of the properties on the map.

![Web application architecture](https://github.com/Linhkust/ML-based-AVM/blob/main/paper%20images/web%20application%20architecture.png)

Users should first initiate `web_application_user.py` to start `Flask` application. Then users can click the web link to `user.html`

```
http://127.0.0.1:5000
```

And it will turn to the web page:

![Web application interface](https://github.com/Linhkust/ML-based-AVM/blob/main/paper%20images/Web%20application%20interface.png)

#### How to use the application?

In the `User Input` interface, you need to select `District`, `Estate`, `Block/Tower`, `Floor level`, `Room`, and `GFA`. Then you click `Google Map` button to find your property location and Google street view around your property. Then you can click `Submit` button to run the automated valuation system to obtain the property valuation.
