import pandas as pd
import datetime
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier


def x_y_split(data, features, predictor):
    X = pd.DataFrame(data, columns=features)
    y = data[predictor]
    return X, y


if __name__ == '__main__':

    # # #

    # Print the start time of the execution
    start = datetime.datetime.now()
    print('Start: ' + str(start))

    # Configure parameters to load the corresponding dataset
    data_type = 'synthetic'  # ['original', 'synthetic']
    if data_type == 'original':
        data_filepath = '../data/'
        data_filename = 'sample.csv'
    elif data_type == 'synthetic':
        data_filepath = '../results/PrivatePreservingDataset/'
        k = 4
        X_1 = 'label'
        epsilon = 0.01
        n_records_generate = 400
        data_filename = 'sample_root_' + X_1 + '_k_' + str(k) + '_epsilon_' + str(epsilon) + \
                        '_n_' + str(n_records_generate) + '.csv'

    # Loading data from file
    print('=======', data_type, '=======')
    data = pd.read_csv(data_filepath + data_filename, header=0)

    loading_time = datetime.datetime.now()
    print('Data Loaded: ' + str(loading_time))

    # Splitting data
    feature_set = data.columns.tolist()
    label = 'label'
    feature_set = [feature for feature in feature_set if feature != label]
    X_train, y_train = x_y_split(data, feature_set, label)

    # # #

    # Model Training on Synthetic Data and Testing on Original Data
    '''
    if data_type == 'synthetic':

        print('### Performance on Original Dataset:')
        data_filepath = '../Data/'
        data_filename = 'sample.csv'
        test_data = pd.read_csv(data_filepath + data_filename, header=0)
        X_test, y_test = x_y_split(data, feature_set, label)

        index = 1
        for model in ['Logistic Regression', 'Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest']:
            print('======= Model ' + str(index) + ': ' + model + ' =======')
            index += 1

            if model == 'Logistic Regression':
                model = LogisticRegression(max_iter=1000)
            elif model == 'Naive Bayes':
                model = ComplementNB(alpha=1.0)
            elif model == 'Support Vector Machine':
                model = svm.SVC(kernel='linear', probability=True)
            elif model == 'Decision Tree':
                model = tree.DecisionTreeClassifier()
            elif model == 'Random Forest':
                model = RandomForestClassifier()
            else:
                print('Undefined model!')
                break

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred))
            print("Recall:", recall_score(y_test, y_pred))
            print("F1 Score:", f1_score(y_test, y_pred))
            y_prob = model.predict_proba(X_test)[:, 1]  # probability for class 1
            print("ROC AUC:", roc_auc_score(y_test, y_prob))
    '''

    # Model Training with Cross Validation
    cv_n = 10
    score_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Logistic Regression
    print('======= Model 1: Logistic Regression =======')
    LR = LogisticRegression(max_iter=1000)
    cv_scores = cross_validate(LR, X_train, y_train, cv=cv_n, scoring=score_list, return_train_score=False)
    for score in score_list:
        print(score, ':', cv_scores['test_' + score].mean())

    # Naive Bayes
    print('======= Model 2: Naive Bayes =======')
    NB = ComplementNB()
    cv_scores = cross_validate(NB, X_train, y_train, cv=cv_n, scoring=score_list, return_train_score=False)
    for score in score_list:
        print(score, ':', cv_scores['test_' + score].mean())

    # Support Vector Machine
    print('======= Model 3: Support Vector Machine =======')
    SVM = svm.SVC(kernel='linear', probability=True)
    cv_scores = cross_validate(SVM, X_train, y_train, cv=cv_n, scoring=score_list, return_train_score=False)
    for score in score_list:
        print(score, ':', cv_scores['test_' + score].mean())

    # Decision Tree
    print('======= Model 4: Decision Tree =======')
    DT = tree.DecisionTreeClassifier()
    cv_scores = cross_validate(DT, X_train, y_train, cv=cv_n, scoring=score_list, return_train_score=False)
    for score in score_list:
        print(score, ':', cv_scores['test_' + score].mean())

    # Random Forest
    print('======= Model 5: Random Forest =======')
    RF = RandomForestClassifier(n_estimators=500)
    cv_scores = cross_validate(RF, X_train, y_train, cv=cv_n, scoring=score_list, return_train_score=False)
    for score in score_list:
        print(score, ':', cv_scores['test_' + score].mean())

    # # #

    # Feature Importance based on RF
    print('======= Feature Importance =======')
    RF = RandomForestClassifier(n_estimators=500)
    RF.fit(X_train, y_train)
    feature_importance = RF.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df)

    # # #
