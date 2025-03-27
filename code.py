### Kaggle - Classification with an Academic Success Dataset
### playground competition
### https://www.kaggle.com/competitions/playground-series-s4e6/overview

print('Librairies')
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, accuracy_score, precision_score, f1_score, recall_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from xgboost import cv
import xgboost as xgb

import time

path = 'B:/GER_ATUARIAL/Dados Estrategicos/Kaggle/Classification with an Academic Success Dataset/'
path_data = path + 'Data/'
path_results = path + 'Results/'

tab = pd.read_csv(path_data + 'train.csv')
test = pd.read_csv(path_data + 'test.csv')

print('Encoding the target variable.')
if True:
    label_encoders = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    reverse_encoders = {v: k for k, v in label_encoders.items()}
    tab['Target'] = tab['Target'].apply(lambda x: label_encoders[x])

print('Correlation matrix.')
if True:
    matrix = tab.corr()
    # Vamos excluir algumas colunas de curricular
    if False:
        curr_remove = [i for i in tab.columns if ('Curricular' in i) and not('approved' in i)]
        tab.drop(columns = curr_remove, inplace = True)
        # Apply to test as well
        test.drop(columns = curr_remove, inplace = True)

print('Adjust data.')
if True:
    # Grouping the occurences with low frequency as Others
    for col in ['Course', 'Marital status', 'Nacionality', 'Application mode', 'Previous qualification', 
                #'Previous qualification (grade)',
                "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]:
        # Group all ... with < 1000 people as 'Others'
        res = tab.groupby([col]).size().reset_index()
        res['New ' + col] = res[col].apply(lambda x : str(x))
        res.loc[res[0] <= 1000, 'New ' + col] = -10
        tab = tab.merge(res[[col, 'New ' + col]], on = col, how = 'left')
        tab.drop(columns = [col], inplace = True)
        tab.rename(columns = {'New ' + col: col}, inplace = True)
        # Apply to test as well
        test = test.merge(res[[col, 'New ' + col]], on = col, how = 'left')
        test.drop(columns = [col], inplace = True)
        test.rename(columns = {'New ' + col: col}, inplace = True)
 
    # Apply the most commun value on the low-frequency occurences
    for col in ['Application order']:
        res = tab.groupby([col]).size().reset_index().sort_values([0],ascending = False).reset_index()
        occ = res.loc[0, col]
        tab.loc[tab[col].isin(res.loc[res[0] < 1000, col].tolist()), col] = occ
        # Apply to test as well
        test.loc[test[col].isin(res.loc[res[0] < 1000, col].tolist()), col] = occ
        test.loc[test[col].isnull(), col] = occ
        
    # Apply the most commun value or mean on missing occurences
    for col in test.columns:
        if len(test.loc[test[col].isnull(), :]) > 0 or len(tab.loc[tab[col].isnull(), :]):
            if (col in ['Admission Grade', 'Age at Enrollment', 'Unemployment rate', 'Inflation rate', 'GDP']) or ('Curricular' in col):
                occ = tab[col].mean() # usamos a media
                tab.loc[tab[col].isnull(), col] = occ
                test.loc[test[col].isnull(), col] = occ
            else:
                res = tab.groupby([col]).size().reset_index().sort_values([0],ascending = False).reset_index()
                occ = res.loc[0, col]
                tab.loc[tab[col].isnull(), col] = occ
                test.loc[test[col].isnull(), col] = occ
    
    if False:
        # Modify some variables : age?
        col = 'Previous qualification'
        res = pd.DataFrame(columns = [col])
        for yval in [0, 1, 2]:
            o = tab.loc[tab['Target'] == yval, :].groupby([col]).size().reset_index().rename(columns = {0: yval})
            res = res.merge(o, on = col, how = 'outer').fillna(0)
        res['Total'] = res[0] + res[1] + res[2]
        res.sort_values(col, ascending = True, inplace = True)
        for yval in [0, 1, 2]:
            res[yval] = res[yval] / (res['Total'])
        
        plt.clf()
        plt.figure()  
        plt.plot(res[col], res[0], label='0')
        plt.plot(res[col], res[1], label='1')
        plt.plot(res[col], res[2], label='2')
        plt.ylim([0.0, 1.05])
        plt.legend()
    
    
    if True:
        def newage(x): # joining all people of more than 25 y.o
            if x<25:
                return x
            else:
                return 25
        tab['Age at enrollment'] = tab['Age at enrollment'].apply(lambda x : newage(x))
        test['Age at enrollment'] = test['Age at enrollment'].apply(lambda x : newage(x))
    
    ### Remove Outliers - for numerical values
    if False:
        import seaborn as sns
        num_col = [i for i in tab.columns if 'rade' in i or 'urricular' in i]
        for c in num_col:
            plt.clf()
            sns.boxplot(x=tab[c], color='lightblue').set_title(c)
            plt.show()
        # no obbious outlier
    
    
    ### Group marital status
    tab.loc[tab['Marital status'].isin([4, 6]), 'Marital status'] = 4 # grouping divorced with legally separated
    
    # Turn multi-class categorical features into dummy variables
    for col in ['Application order', 'Course', 'Marital status', 'Application mode', 'Previous qualification',
                "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]:
        for value in tab[col].unique().tolist()[:-1]:
            tab[col + ' - ' + str(value)] = 0
            tab.loc[tab[col] == value, col + ' - ' + str(value)] = 1
            test[col + ' - ' + str(value)] = 0
            test.loc[test[col] == value, col + ' - ' + str(value)] = 1
        tab.drop(columns = [col], inplace = True)   
        test.drop(columns = [col], inplace = True)

print('Feature selection, and data preparation.')
if True:
    features = [i for i in tab.columns if i not in ['Target', 'id']]
    x, y = tab[features], tab['Target']
    submissions = test[features]
    
    # standard scaler
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    submissions = scaler.transform(submissions)
    
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    print('Proportions:', y_train.value_counts(1))
    
   
''' ---- Models ---- '''
print('--- Models ---')

# Dummy model
if False:
    y_dummy = [2 for i in range(1, len(y_test) + 1)]
    acc = accuracy_score(y_dummy, y_test)
    print('Dummy accuracy:', round(100 * acc), '%')


# Logistic Regression
if False:
    log = LogisticRegression(penalty = 'l2')
    log.fit(x_train, y_train)
    y_log = log.predict(x_test)
    probs = log.predict_proba(x_test)
    res = pd.DataFrame()
    #res['Prob 0'], res['Prob 1'], res['Prob 2'] = probs[:, 0], probs[:, 1], probs[:, 2]
    res['ytest'], res['ylog'] = y_test, y_log
    res['ytest'].value_counts(1), res['ylog'].value_counts(1)
    res['Result'] = 0
    res.loc[res['ytest'] == res['ylog'], 'Result'] = 1
    acc_test = accuracy_score(y_log, y_test)
    print('Logistic Regression accuracy Test:', round(100 * acc_test, 2), '%')
    acc_train = accuracy_score(log.predict(x_train), y_train)
    print('Logistic Regression accuracy Train:', round(100 * acc_train, 2), '%')

    # What is the accuracy per class? Good on 0 and 2 but bad on 1.
    res.groupby(['ytest'])['Result'].mean().reset_index()
        
    # Predict class for submission file
    sub_preds = log.predict(submissions)
    res = pd.DataFrame()
    res['id'] = test['id'].tolist()
    res['Target'] = [reverse_encoders[i] for i in sub_preds]
    res.to_csv(path_results + 'Model 1 - Logistic Regression - submissions.csv', index = False)
    
    
# GLM

# Gradient Boosting
if True:
    t1 = time.time()
    param_grid = {
        'n_estimators': [600, 650, 700],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.04, 0.05, 0.06],
        'subsample': [0.7, 0.75],
        'colsample_bytree': [0.4, 0.5, 0.6],
        'min_child_weight': [1, 3, 5, 7, 9, 10, 12],
        }
    
    # Grid Search
    if False:
        xgb_classifier = XGBClassifier()
        print('Grid Search com criteria accuracy')
        grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='accuracy', n_jobs = 5)
        grid_search.fit(x_train, y_train)
        res = pd.DataFrame(grid_search.cv_results_)
        t2 = time.time()
        print('Time to the gridsearch:', round((t2 - t1)/60, 1), 'min')
        print('Best hyperparameters: ', grid_search.best_params_)
        print('Best accuracy score:', round(grid_search.best_score_, 4))
        xgb_classifier = grid_search.best_estimator_
        if False:
            acc_test = accuracy_score(xgb_classifier.predict(x_test), y_test)
            print('xgb accuracy Test:', round(100 * acc_test, 1), '%')
            acc_train = accuracy_score(xgb_classifier.predict(x_train), y_train)
            print('xgb accuracy Train:', round(100 * acc_train, 1), '%')
            scores = cross_val_score(xgb_classifier, x_train, y_train, cv = 5)
            print('Cross val score train', round(100 * scores.mean(), 2), '%')

    if False:
        # Make first submission
        sub_preds = xgb_classifier.predict(submissions)
        res = pd.DataFrame()
        res['id'] = test['id'].tolist()
        res['Target'] = [reverse_encoders[i] for i in sub_preds]
        res.to_csv(path_results + 'Model 2- xgb - submissions.csv', index = False)
    
        # Tune in detail each parameter, by fixing the others.
        
        bestparams = grid_search.best_params_
        max_depth = bestparams['max_depth']
        learning_rate = bestparams['learning_rate']
        n_estimators = bestparams['n_estimators']
        subsample = bestparams['subsample']   
    
        bestparams = grid_search.best_params_
        max_depth = bestparams['max_depth']
        learning_rate = bestparams['learning_rate']
        n_estimators = bestparams['n_estimators']
        subsample = bestparams['subsample']    
        
        # Learning rate
        res = pd.DataFrame(columns = ['Learning Rate', 'Accuracy train', 'Accuracy test'])
        lrs, score_train, score_test = [], [], []
        for lr in np.arange(0.001, 1.0, 0.001):
            lrs.append(lr)
            xgb_classifier = XGBClassifier(learning_rate = lr, max_depth = max_depth, n_estimators = n_estimators, subsample = subsample)
            xgb_classifier.fit(x_train, y_train)
            score_test.append(accuracy_score(xgb_classifier.predict(x_test), y_test))
            score_train.append(accuracy_score(xgb_classifier.predict(x_train), y_train))
        res['Learning Rate'], res['Accuracy train'], res['Accuracy test'] = lrs, score_train, score_test
        print('Train:', round(res['Accuracy train'].max(), 4))
        print('Test:', round(res['Accuracy test'].max(), 4))        
        
        plt.clf()
        plt.figure()  
        plt.plot(res['Learning Rate'], res['Accuracy train'], label='train')
        plt.plot(res['Learning Rate'], res['Accuracy test'], label='test')
        plt.ylim([0.78, 0.9])
        plt.legend()
        
        learning_rate = 0.2
        
        # n_estimators
        res = pd.DataFrame(columns = ['n_estimators', 'Accuracy train', 'Accuracy test'])
        lrs, score_train, score_test = [], [], []
        for lr in np.arange(300, 500, 10):
            print(lr)
            lrs.append(lr)
            xgb_classifier = XGBClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = lr, subsample = subsample)
            xgb_classifier.fit(x_train, y_train)
            score_test.append(accuracy_score(xgb_classifier.predict(x_test), y_test))
            score_train.append(accuracy_score(xgb_classifier.predict(x_train), y_train))
        res['n_estimators'], res['Accuracy train'], res['Accuracy test'] = lrs, score_train, score_test
        print('Train:', round(res['Accuracy train'].max(), 4))
        print('Test:', round(res['Accuracy test'].max(), 4))        
        
        plt.clf()
        plt.figure()  
        plt.plot(res['n_estimators'], res['Accuracy train'], label='train')
        plt.plot(res['n_estimators'], res['Accuracy test'], label='test')
        plt.ylim([0.78, 0.9])
        plt.legend()
        
        n_estimators = 200
        
        # subsample
        res = pd.DataFrame(columns = ['subsample', 'Accuracy train', 'Accuracy test'])
        lrs, score_train, score_test = [], [], []
        for lr in np.arange(0.1, 1, 0.1):
            print(lr)
            lrs.append(lr)
            xgb_classifier = XGBClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators, subsample = lr)
            xgb_classifier.fit(x_train, y_train)
            score_test.append(accuracy_score(xgb_classifier.predict(x_test), y_test))
            score_train.append(accuracy_score(xgb_classifier.predict(x_train), y_train))
        res['subsample'], res['Accuracy train'], res['Accuracy test'] = lrs, score_train, score_test
        print('Train:', round(res['Accuracy train'].max(), 4))
        print('Test:', round(res['Accuracy test'].max(), 4))        
        
        plt.clf()
        plt.figure()  
        plt.plot(res['subsample'], res['Accuracy train'], label='train')
        plt.plot(res['subsample'], res['Accuracy test'], label='test')
        plt.ylim([0.78, 0.9])
        plt.legend()
        subsample = 0.8
        
    n_estimators = 500 # 500
    subsample = 0.7 # 0.7
    learning_rate = 0.05 # 0.05
    max_depth = 5 # 5
    xgb_classifier = XGBClassifier(n_estimators = n_estimators, subsample = subsample, max_depth = max_depth, learning_rate = learning_rate)
    xgb_classifier.fit(x_train, y_train)
    acc_test = accuracy_score(xgb_classifier.predict(x_test), y_test)
    print('xgb accuracy Test:', round(100 * acc_test, 2), '%')
    acc_train = accuracy_score(xgb_classifier.predict(x_train), y_train)
    print('xgb accuracy Train:', round(100 * acc_train, 2), '%')
    scores = cross_val_score(xgb_classifier, x_train, y_train, cv = 5)
    print('Cross val score train', round(100 * scores.mean(), 2), '%')
    
    feature_imp = pd.DataFrame()
    feature_imp['Features'] = features
    feature_imp['Features Importance'] = xgb_classifier.feature_importances_
    # tuning parameters based on cross val score train

    
    if True:
        # Make xgb submission
        xgb_classifier = XGBClassifier(n_estimators = n_estimators, subsample = subsample, max_depth = max_depth, learning_rate = learning_rate)
        new_x = np.concatenate((x_train, x_test), axis = 0)
        new_y = np.concatenate((y_train, y_test), axis = 0)
        #new_x = x_train
        #new_y = y_train
        xgb_classifier.fit(new_x, new_y)
        acc_test = accuracy_score(xgb_classifier.predict(x_test), y_test)
        print('xgb accuracy Test:', round(100 * acc_test, 2), '%')
        acc_train = accuracy_score(xgb_classifier.predict(x_train), y_train)
        print('xgb accuracy Train:', round(100 * acc_train, 2), '%')
      
        sub_preds = xgb_classifier.predict(submissions)
        res = pd.DataFrame()
        res['id'] = test['id'].tolist()
        res['Target'] = [reverse_encoders[i] for i in sub_preds]
        res.to_csv(path_results + 'Model 4 - xgb new - submissions.csv', index = False)


# Random Forest
if False:
    print('Random Forest')
    criterion = 'gini' # entropy, log_loss
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    min_samples_leaf = 10
    
    forest = RandomForestClassifier(
        criterion = criterion,
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf)
    forest.fit(x_train, y_train)
    
    acc_test = accuracy_score(forest.predict(x_test), y_test)
    print('forest accuracy Test:', round(100 * acc_test, 2), '%')
    acc_train = accuracy_score(forest.predict(x_train), y_train)
    print('forest accuracy Train:', round(100 * acc_train, 2), '%')
    
    # Grid Search
    if False:
        t1 = time.time()
        param_grid = {
            'n_estimators': [80, 120, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf':[5, 10, 20]
            }
        forest = RandomForestClassifier()
        print('Grid Search com criteria accuracy')
        grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='accuracy', n_jobs = 5)
        grid_search.fit(x_train, y_train)
        res = pd.DataFrame(grid_search.cv_results_)
        t2 = time.time()
        print('Time to the gridsearch:', round((t2 - t1)/60, 1), 'min')
        print('Best hyperparameters: ', grid_search.best_params_)
        print('Best accuracy score:', round(grid_search.best_score_, 4))
    
    forest = RandomForestClassifier(
        criterion = 'entropy',
        max_depth = 15, min_samples_leaf = 5,
        min_samples_split = 10, n_estimators = 300)
    forest.fit(x_train, y_train)
    acc_test = accuracy_score(forest.predict(x_test), y_test)
    print('forest accuracy Test:', round(100 * acc_test, 2), '%')
    acc_train = accuracy_score(forest.predict(x_train), y_train)
    print('forest accuracy Train:', round(100 * acc_train, 2), '%')
    # Model opti
    xgb_params = {'grow_policy': 'depthwise', 'tree_method': 'hist', 'enable_categorical': True, 'gamma': 0, 'n_estimators': 768, 'learning_rate': 0.026111403303690425, 'max_depth': 8, 'reg_lambda': 26.648168065161098, 'min_child_weight': 1.0626186255116183, 'subsample': 0.8580490989206254, 'colsample_bytree': 0.5125814118774029}
    model_XGBop = XGBClassifier(**xgb_params)
    model_XGBop.fit(x_train, y_train)
    acc_test = accuracy_score(model_XGBop.predict(x_test), y_test)
    print('xgb accuracy Test:', round(100 * acc_test, 2), '%')
    acc_train = accuracy_score(model_XGBop.predict(x_train), y_train)
    print('xgb accuracy Train:', round(100 * acc_train, 2), '%')
    scores = cross_val_score(model_XGBop, x_train, y_train, cv = 5)
    print('Cross val score train', round(100 * scores.mean(), 2), ' %')
 

# Regarder des exemples de oú le modele se trompe

# Regarder, selon les variables, l'accuracy.










