# Load libraries
# from scipy.stats.morestats import stats
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.datasets import make_classification
from numpy import std
from numpy import mean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as pl

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
import statistics as stat
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
# from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.pipeline import Pipeline
import shap
import os
from sklearn.preprocessing import MinMaxScaler
import sklearn.externals as extjoblib
import joblib


def plot_feature_importance(model, feature, name, figure_path):
    path = './ml_feature_importance/'
    if name == 'RF' or name == 'LGBM':
        importances = model.estimator_.feature_importances_
    elif name == 'SVM':
        importances = model.ranking_
    elif name == 'MLR':
        coefficients = model.estimator_.coef_
        importances = np.mean(np.abs(coefficients), axis=0)
    # print("importances", importances)
    # Sort feature importances in descending order
    # sorted_indices = importances.argsort()[::-1]
    # sorted_importances = importances[sorted_indices]
    # print("sorted_indices", sorted_indices)
    # print("sorted_importances", sorted_importances)
    # Get the names of the selected features
    selected_feature_names = [feature[i]
                              for i in model.get_support(indices=True)]
    sorted_lists = sorted(
        zip(importances, selected_feature_names), reverse=True)
    sorted_feature_scores, sorted_feature_names = zip(*sorted_lists)
    # print(len(selected_feature_names), selected_feature_names)
    # print(sorted_feature_scores, sorted_feature_names)

    # Plot the feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_feature_scores)),
            sorted_feature_scores, tick_label=sorted_feature_names)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance (RFE)')
    plt.subplots_adjust(bottom=0.4)
    # plt.show()
    plt.savefig(os.path.join(path, figure_path + name + '.png'))


def Decision_Tree(X, y, train_group, case):

    if case == 2:
        model = DecisionTreeClassifier(
            max_depth=None, criterion='entropy', class_weight='balanced', random_state=0)

        # RandomizedSearchCV with straightfied group kfold.
        parameters = {'max_depth': range(3, 10, 1),
                      'criterion': ['gini', 'entropy'],
                      'class_weight': ['balanced', None, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 5}, {0: 1, 1: 2, 2: 3}],
                      'ccp_alpha': [0, 1, 3],
                      'random_state': [1, 5, 7]}
        DT = RandomizedSearchCV(model, parameters, scoring='accuracy', cv=4, n_iter=100,
                                refit=True, n_jobs=1, verbose=9, return_train_score=True,
                                random_state=0, error_score='raise')
        scores = DT.fit(X, y, groups=train_group)
        print(scores.best_score_)
        return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型

    # print(confusion_matrix(y_test, y_predict))
    if case == 1:
        model = train_group
        clf = model.fit(X, y)
        return clf


def Random_Forest(X, y, mod, case, groups, figure_path):

    if case == 2:
        model = RandomForestClassifier(max_depth=None, n_jobs=-1)
        selector = RFE(model, n_features_to_select=None, step=1, verbose=9)
        # RandomizedSearchCV with straightfied group kfold.
        parameters = {'max_depth': range(3, 20, 2),
                      'n_estimators': [100, 300, 500],
                      'criterion': ['gini', 'entropy'],
                      'class_weight': ['balanced', None, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 5}, {0: 1, 1: 2, 2: 3}],
                      'ccp_alpha': [0, 1, 3],
                      'random_state': [1, 5, 7]}
        RF = RandomizedSearchCV(selector, parameters, scoring='accuracy', cv=8,
                                refit=True, n_jobs=-1, verbose=9, return_train_score=True,
                                random_state=0)

        RF.fit(X, y, groups=groups)
        print(RF.best_score_)

        return RF.best_estimator_
        # return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型

        # best_estimator = feature_selection_RFE(RF, X, y)
        # return best_estimator

    elif case == 1:
        clf = mod.fit(X, y)
        return clf

    elif case == 3:
        model = mod
        selector = RFE(model, n_features_to_select=20, step=1, verbose=0)
        selector.fit(X, y)
        plot_feature_importance(selector, X.columns, 'RF', figure_path)
        return selector
        # model, features, ranking = feature_selection_RFE(model, X, y)


def KNN(X, y, train_group, case):

    if case == 2:
        model = KNeighborsClassifier(n_jobs=-1)

        parameters = {'n_neighbors': range(3, 20, 2),
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], }
        KNN = RandomizedSearchCV(model, parameters, scoring='accuracy', cv=4,
                                 refit=True, n_jobs=-1, verbose=9, return_train_score=True,
                                 random_state=0)

        scores = KNN.fit(X, y, groups=train_group)
        print(scores.best_score_)
        return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型

    if case == 1:
        model = train_group
        clf = model.fit(X, y)
        return clf


def Naive_Bayes(X, y, train_group, case):

    if case == 2:
        model = GaussianNB()

        parameters = {'var_smoothing': [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]}
        NB = RandomizedSearchCV(model, parameters, scoring='accuracy', cv=4,
                                refit=True, n_jobs=-1, verbose=9, return_train_score=True,
                                random_state=0)

        scores = NB.fit(X, y, groups=train_group)
        print(scores.best_score_)
        return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型

    if case == 1:
        model = train_group
        clf = model.fit(X, y)
        return clf


def SVM(X, y, mod, case, groups, figure_path):

    if case == 2:
        model = svm.SVC(C=1, max_iter=-1, decision_function_shape='ovo')

        parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'tol': [1e-3, 1e-5, 1e-7, 1e-9],
                      'class_weight': ['balanced', None, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 5}, {0: 1, 1: 2, 2: 3}],
                      'random_state': [1, 5, 7]}
        Svm = RandomizedSearchCV(model, parameters, scoring='accuracy', cv=4,
                                 refit=True, n_jobs=-1, verbose=9, return_train_score=True,
                                 random_state=0)

        # scores = Svm.fit(X, y, groups=groups)
        # print(scores.best_score_)
        # return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型
        best_estimator = feature_selection_RFE(Svm, X, y)
        return best_estimator

    elif case == 1:
        clf = mod.fit(X, y)
        return clf

    elif case == 3:
        model = mod
        selector = RFE(model, n_features_to_select=20, step=1, verbose=0)
        selector.fit(X, y)
        plot_feature_importance(selector, X.columns, 'SVM', figure_path)
        return selector


def Multinomial_LR(X, y, mod, case, groups, figure_path):

    if case == 2:
        mul_logreg = LogisticRegression(
            C=1, multi_class='multinomial', solver='lbfgs', n_jobs=-1)

        parameters = {'penalty': ['l2', 'none'],
                      'tol': [1e-3, 1e-5, 1e-7, 1e-9],
                      'class_weight': ['balanced', None, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 5}, {0: 1, 1: 2, 2: 3}],
                      'random_state': [1, 5, 7],
                      'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                      'max_iter': [100, 300, 500]}
        MLR = RandomizedSearchCV(mul_logreg, parameters, scoring='accuracy', cv=4,
                                 refit=True, n_jobs=-1, verbose=0, return_train_score=True,
                                 random_state=0)

        # scores = MLR.fit(X, y, groups=groups)
        # print(scores.best_score_)
        # return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型
        best_estimator = feature_selection_RFE(MLR, X, y)
        return best_estimator

    elif case == 1:
        mul_logreg = mod.fit(X, y)
        return mul_logreg

    elif case == 3:
        model = mod
        selector = RFE(model, n_features_to_select=20, step=1, verbose=0)
        selector.fit(X, y)
        plot_feature_importance(selector, X.columns, 'MLR', figure_path)
        return selector


def Adaboost(X, y, train_group, case):

    if case == 2:
        base_estimator = AdaBoostClassifier(DecisionTreeClassifier(ccp_alpha=0, class_weight='balanced', max_depth=5,
                                                                   random_state=7))

        parameters = {'n_estimators': [100, 300, 500],
                      'learning_rate': [1e-3, 1e-5, 1e-7, 1e-9],
                      'algorithm': ['SAMME', 'SAMME.R'],
                      'random_state': [1, 5, 7]}
        Ada = RandomizedSearchCV(base_estimator, parameters, scoring='accuracy', cv=4,
                                 refit=True, n_jobs=-1, verbose=9, return_train_score=True,
                                 random_state=0)

        scores = Ada.fit(X, y, groups=train_group)
        print(scores.best_score_)
        return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型

    if case == 1:
        model = train_group
        clf = model.fit(X, y)
        return clf


def Xgboost(X, y, train_group, case):

    if case == 2:
        model = XGBClassifier(objective='multi:softmax')

        parameters = {'eta': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
                      'max_depth': range(3, 10, 1),
                      'subsample': [0.5, 0.7, 0.9],
                      'eval_metric': ['merror', 'mlogloss']}
        xgb = RandomizedSearchCV(model, parameters, scoring='accuracy', cv=4,
                                 refit=True, n_jobs=-1, verbose=1, return_train_score=True,
                                 random_state=0)

        scores = xgb.fit(X, y, groups=train_group)
        print(scores.best_score_)
        return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型

    if case == 1:
        model = train_group
        clf = model.fit(X, y)
        return clf


def LightGBM(X, y, mod, case, groups, figure_path):

    if case == 2:
        model = LGBMClassifier(objective='multiclass',
                               n_jobs=-1, boosting_type='goss')

        parameters = {'boosting_type': ['gbdt', 'dart', 'goss'],
                      'max_depth': [3, 5, 7, 9],
                      'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
                      'n_estimators': [100, 300, 500],
                      'class_weight': ['balanced', None, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 5}, {0: 1, 1: 2, 2: 3}],
                      'reg_alpha': [0, 0.1, 0.01, 0.001, 0.0001],
                      'reg_lambda': [0, 0.1, 0.01, 0.001, 0.0001]}
        # 'boosting_type': ['gbdt', 'dart', 'rf'],
        Lgbm = RandomizedSearchCV(model, parameters, scoring='accuracy', cv=4,
                                  refit=True, n_jobs=-1, verbose=9, return_train_score=True,
                                  random_state=0, error_score='raise')

        # scores = Lgbm.fit(X, y, groups=groups)
        # print(scores.best_score_)
        # return scores.best_estimator_, scores.best_score_  # 回傳最佳參數模型
        best_estimator = feature_selection_RFE(Lgbm, X, y)
        return best_estimator

    elif case == 1:
        clf = mod.fit(X, y)
        return clf

    elif case == 3:
        model = mod
        selector = RFE(model, n_features_to_select=20, step=1, verbose=0)
        selector.fit(X, y)
        plot_feature_importance(selector, X.columns, 'LGBM', figure_path)
        return selector


def Dictionary_to_list(report):
    p0 = report['0']['precision']
    p1 = report['1']['precision']
    # p2 = report['2']['precision']
    r0 = report['0']['recall']
    r1 = report['1']['recall']
    # r2 = report['2']['recall']
    f0 = report['0']['f1-score']
    f1 = report['1']['f1-score']
    # f2 = report['2']['f1-score']
    acc = report['accuracy']
    mp = report['macro avg']['precision']
    mr = report['macro avg']['recall']
    mf = report['macro avg']['f1-score']
    wp = report['weighted avg']['precision']
    wr = report['weighted avg']['recall']
    wf = report['weighted avg']['f1-score']
    return [p0, p1, r0, r1, f0, f1, acc, mp, mr, mf, wp, wr, wf]


def model_predict(model, name, X, y, fold, figure_path):
    path = './results/cm/'
    predict = model.predict(X)
    report = classification_report(y, predict, output_dict=True)
    report = Dictionary_to_list(report)
    temp = [name]
    temp.extend(fold)
    temp.extend(report)
    print(temp)

    # confusion matrix
    # class_names = ['Warthin\'s', 'PA', 'Malignant']  # Replace with your actual class names
    # plot_confusion_matrix(model, X, y, display_labels=class_names)
    # plt.title('Confusion Matrix')
    # plt.savefig(os.path.join(path, figure_path + name + '.png'))

    return temp


def feature_importance_SHAP(model, X, name, Feature, figure_path, feature_type):
    path = './results/ml_shap_plot/'
    temp = []
    # statistic = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(model.support_)):
        if (model.support_[i] == True):
            temp.append(model.feature_names_in_[i])
            # statistic[feature_type[model.feature_names_in_[i]]] += 1
    X = X[temp]
    if (name == 'SVM' or name == 'MLR'):
        explainer = shap.LinearExplainer(model.estimator_, X)
        shap_values = explainer.shap_values(X)
    elif (name == 'RF'):
        explainer = shap.TreeExplainer(model.estimator_, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
    elif (name == 'LGBM'):
        explainer = shap.explainers.Tree(model.estimator_, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
    # print(temp)
    # shap_values = explainer.shap_values(X)
    plt.title(name + ' model SHAP Values', fontsize=15)
    shap.summary_plot(shap_values, X, feature_names=temp, max_display=len(temp),
                      show=False, class_names=['Warthin', 'PA', 'Malignant'],
                      class_inds='original', color=pl.get_cmap("Paired"))
    plt.savefig(os.path.join(path, figure_path + name + '.png'))
    plt.clf()
    # print(statistic)


def RFE_feature(selector, Feature, model):
    list = [model]
    list.extend(selector.ranking_)
    return list


def CompareModel(df, Feature, run_num):
    '''run five round that the class in train test is balanced'''

    seed = range(0, 10000, 2)
    count = 0
    result = []
    RF_acc, SVM_acc, MLR_acc, LGBM_acc = 0, 0, 0, 0
    RF_final_model, SVM_final_model, MLR_final_model, LGBM_final_model = 0, 0, 0, 0
    RF_scaler, SVM_scaler, MLR_scaler, LGBM_scaler = 0, 0, 0, 0
    RF_X, SVM_X, MLR_X, LGBM_X = 0, 0, 0, 0
    select_feature_dict = {}
    feature_type = {}
    # for i in range(0,6):
    #     feature_type[Feature[i]] = 0
    # for i in range(6,102):
    #     feature_type[Feature[i]] = 1
    # for i in range(102,126):
    #     feature_type[Feature[i]] = 1
    # for i in range(126,142):
    #     feature_type[Feature[i]] = 2
    # for i in range(142,158):
    #     feature_type[Feature[i]] = 3
    # for i in range(158,163):
    #     feature_type[Feature[i]] = 4
    # for i in range(163,177):
    #     feature_type[Feature[i]] = 5
    # for i in range(179,184):
    #     feature_type[Feature[i]] = 0
    # feature_type['age'] = 6
    # feature_type['sex'] = 6
    for s in seed:

        if count >= 1:  # 跑100次
            break

        train, test = train_test_split(df, test_size=0.2, random_state=s)

        X_train, X_test, y_train, y_test = train[Feature], test[Feature], train['Maligant'], test['Maligant']
        train_group = np.array([])

        scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = pd.DataFrame(X_train)
        X_train.columns = Feature

        X_test = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test)
        X_test.columns = Feature

        c1, c2, c3 = 0, 0, 0
        for i in y_test:
            if i == 0:
                c1 += 1
            elif i == 1:
                c2 += 1
            elif i == 2:
                c3 += 1

        rc1 = c1 / (c1+c2+c3)
        rc2 = c2 / (c1+c2+c3)
        rc3 = c3 / (c1+c2+c3)

        fold = [c1, c2, c3]
        c1, c2, c3 = 0, 0, 0
        for i in y_train:
            if i == 0:
                c1 += 1
            elif i == 1:
                c2 += 1
            elif i == 2:
                c3 += 1
        # if c1!=88 or c2!=97 or c3!=85:
        #     continue
        count += 1
        fold.extend([c1, c2, c3])

        figure_path = '0711_' + str(c1) + "_" + str(c2) + "_" + str(c3) + "_"

        RF_model = RandomForestClassifier(
            max_depth=5, n_jobs=-1, criterion='entropy', class_weight='balanced', random_state=0)
        RF_model = Random_Forest(
            X_train, y_train, RF_model, run_num, train_group, figure_path)
        temp = model_predict(RF_model, 'RF', X_test, y_test, fold, figure_path)
        result.append(temp)
        if (temp[13] >= RF_acc):
            RF_final_model = RF_model
            RF_scaler = scaler
            RF_X = pd.DataFrame(X_train, columns=Feature)

        SVM_model = svm.SVC(
            C=1, max_iter=-1, decision_function_shape='ovo', class_weight=None, kernel='linear')
        SVM_model = SVM(X_train, y_train, SVM_model,
                        run_num, train_group, figure_path)
        temp = model_predict(SVM_model, 'SVM', X_test,
                             y_test, fold, figure_path)
        result.append(temp)
        if (temp[13] >= SVM_acc):
            SVM_final_model = SVM_model
            SVM_scaler = scaler
            SVM_X = pd.DataFrame(X_train, columns=Feature)

        MLR_model = LogisticRegression(
            C=1, max_iter=300, multi_class='multinomial', n_jobs=-1, solver='saga', tol=0.001)
        MLR_model = Multinomial_LR(
            X_train, y_train, MLR_model, run_num, train_group, figure_path)
        temp = model_predict(MLR_model, 'MLR', X_test,
                             y_test, fold, figure_path)
        result.append(temp)
        if (temp[13] >= MLR_acc):
            MLR_final_model = MLR_model
            MLR_scaler = scaler
            MLR_X = pd.DataFrame(X_train, columns=Feature)

        Lgbm_model = LGBMClassifier(boosting_type='gbdt', class_weight='balanced', max_depth=5,
                                    n_estimators=100, objective='binary', reg_alpha=0.1,
                                    reg_lambda=0.1, force_col_wise=True, num_leaves=5)
        Lgbm_model = LightGBM(X_train, y_train, Lgbm_model,
                              run_num, train_group, figure_path)
        temp = model_predict(Lgbm_model, 'Lgbm', X_test,
                             y_test, fold, figure_path)
        result.append(temp)
        if (temp[13] >= LGBM_acc):
            LGBM_final_model = Lgbm_model
            LGBM_scaler = scaler
            LGBM_X = pd.DataFrame(X_train, columns=Feature)

        if run_num == 3:

            '''特徵重要性視覺化圖表'''
            figure_path = '0711_' + \
                str(c1) + "_" + str(c2) + "_" + str(c3) + "_"
            feature_importance_SHAP(
                RF_model, X_train, 'RF', Feature, figure_path, feature_type)
            feature_importance_SHAP(
                SVM_model, X_train, 'SVM', Feature, figure_path, feature_type)
            feature_importance_SHAP(
                MLR_model, X_train, 'MLR', Feature, figure_path, feature_type)
            feature_importance_SHAP(
                Lgbm_model, X_train, 'LGBM', Feature, figure_path, feature_type)

    """save model weight"""
    md = [RF_final_model, SVM_final_model, MLR_final_model, LGBM_final_model]

    model_savepath = './results/ml_best_model/'
    # joblib.dump(RF_final_model, model_savepath + 'RF.pkl')
    # joblib.dump(SVM_final_model,model_savepath + 'SVM.pkl')
    # joblib.dump(LGBM_final_model,model_savepath + 'LGBM.pkl')
    # joblib.dump(MLR_final_model,model_savepath + 'MLR.pkl')
    # joblib.dump(RF_scaler, model_savepath + 'RF_scaler.pkl')
    # joblib.dump(SVM_scaler, model_savepath + 'SVM_scaler.pkl')
    # joblib.dump(MLR_scaler, model_savepath + 'MLR_scaler.pkl')
    # joblib.dump(LGBM_scaler,model_savepath + 'LGBM_scaler.pkl')

    return result


def TOCSV(fold, name, report):
    temp = fold.copy()
    temp.extend([name])
    temp.extend([float(report['0']['precision']), float(
        report['0']['recall']), float(report['0']['f1-score'])])
    temp.extend([report['1']['precision'], report['1']
                ['recall'], report['1']['f1-score']])
    temp.extend([report['2']['precision'], report['2']
                ['recall'], report['2']['f1-score']])
    temp.extend([report['accuracy'], report['weighted avg']['precision'],
                report['weighted avg']['recall'], report['weighted avg']['f1-score']])
    print(temp)
    return temp


def feature_selection_RFE(estimator, X, y):
    selector = RFE(estimator, n_features_to_select=None, step=1, verbose=9)
    selector.fit(X, y)
    # rfe_score = []
    # for i in selector.cv_results_:
    #     rfe_score.append(selector.cv_results_[i])
    # rfe_score = [sum(x) for x in zip(*rfecv_score)]

    selected_feature = []
    selected_feature_rank = []
    for i in range(len(selector.support_)):
        if selector.support_[i] == True:
            selected_feature.append(selector.feature_names_in_[i])  # 儲存有被選到的特徵
            selected_feature_rank.append(selector.ranking_[i])  # 儲存有被選到的特徵分數
    return selector, selected_feature, selected_feature_rank


def Pca_down_dimension(df, Feature):

    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.99, svd_solver='full', random_state=0)
    pca.fit(df[Feature])
    X_pca = pca.transform(df[Feature])
    print("original shape:   ", df[Feature].shape)
    print("transformed shape:", X_pca.shape)
    df2 = pd.DataFrame(X_pca)
    Feature = df2.columns.values
    # df2['patient'] = df['patient']
    df2['Maligant'] = df['Maligant']
    print(Feature)
    print(df2)
    return df2, Feature


if __name__ == "__main__":
    # 讀取檔案分類csv
    file_name = './output.csv'  # File name
    df = pd.read_csv(file_name)

    # # 新增病人欄位
    # df["patient"] = np.nan
    # ydict = {}
    # now = 0
    # for i in range(len(df)):
    #     temp = df.loc[i, 'filename'].split(",")
    #     temp = df.loc[i, 'filename'].split(" ")
    #     if temp[0] in ydict.keys():
    #         df.loc[i, 'patient'] = ydict[temp[0]]
    #     else:
    #         ydict[temp[0]] = int(now)
    #         now += 1
    #         df.loc[i, 'patient'] = ydict[temp[0]]

    '''
    0:filename, 1:calss, 2-7:morphological, 8-104:GLCM, 104-129:LBP, 130-205:5GL, 206-210:morphological, 211:patient
    # 獲取feature column
    '''
    Feature = list(df.columns)

    # m1, f1, m2, f2, m3, f3 = 0,0,0,0,0,0
    # for i in range(len(df)):
    #     if df.loc[i, 'class'] == 0:
    #         if df.loc[i, 'sex'] == 1:
    #             m1+=1
    #         else:
    #             f1+=1
    #     elif df.loc[i, 'class'] == 1:
    #         if df.loc[i, 'sex'] == 1:
    #             m2+=1
    #         else:
    #             f2+=1
    #     elif df.loc[i, 'class'] == 2:
    #         if df.loc[i, 'sex'] == 1:
    #             m3+=1
    #         else:
    #             f3+=1

    # print(round(np.mean(m1), 2), '===', round(np.std(m1), 2), '===', len(m1))
    # print(round(np.mean(m2), 2), '===', round(np.std(m2), 2), '===', len(m2))
    # print(round(np.mean(m3), 2), '===', round(np.std(m3), 2), '===', len(m3))
    # print(round(np.mean(f1), 2), '===', round(np.std(f1), 2), '===', len(f1))
    # print(round(np.mean(f2), 2), '===', round(np.std(f2), 2), '===', len(f2))
    # print(round(np.mean(f3), 2), '===', round(np.std(f3), 2), '===', len(f3))
    # print(Feature)
    # print(m1, f1)
    # print(m2, f2)
    # print(m3, f3)

    # PCA降維
    # df, Feature = Pca_down_dimension(df, Feature)
    # X = df[Feature]
    # y = df['class']

    answer = []
    seed = list(range(1, 101))
    seed = [0, 5, 7]
    # train_group = df['patient'].tolist()  # 以patient之影像group為單位

    col = ['model', 'test1', 'test2', 'test3', 'train1', 'train2', 'train3', 'p0', 'p1',
           'r0', 'r1', 'f0', 'f1', 'acc', 'mp', 'mr', 'mf', 'wp', 'wr', 'wf']

    '''多次訓練比較模型成效'''
    run_num = 3  # 1: traing 1次, 2: 超參數, 3: RFE特徵選擇
    compare_result = CompareModel(df, Feature, run_num)
    base_model_result_df = pd.DataFrame(compare_result, columns=col)
    base_model_result_df = base_model_result_df.sort_values(by=['model'])
    print(base_model_result_df)
    base_model_result_df.to_csv(
        './data/base_9model_RFE_imgROI_result_0711.csv', index=False)
