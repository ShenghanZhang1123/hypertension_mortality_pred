from arg_tools import str2bool
from dataloader import label_balancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import xgboost
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from visualization import roc_curves_plot
import numpy as np
import matplotlib.pyplot as plt
import shap
import argparse
import warnings
import pandas as pd
import sys
import os
#os.chdir('./EHR_Fairness/Mimic')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", type=str, default='../Data_iv/new_var')
parser.add_argument("--file-prefix", type=str, default='complication_chart_lab_drug_')
parser.add_argument("--disease", type=str, default='hypertension')
parser.add_argument("--control", type=str, default='')
parser.add_argument("--icd-format", type=str, default='CCS')
parser.add_argument("--ATC-level", type=int, default=4)
parser.add_argument("--task", type=str, default='mortality')
parser.add_argument("--mimic-iv", type=str2bool, default=True)
parser.add_argument("--label-balance", type=str2bool, default=True)
parser.add_argument("--test-balance", type=str2bool, default=True)
parser.add_argument("--icustay", type=str2bool, default=False)

parser.add_argument("--balance-method", type=str, default='down')
parser.add_argument("--algorithm-balance", type=str2bool, default=False)
parser.add_argument("--gamma", type=float, default=10)

parser.add_argument("--top", type=int, default=30)
parser.add_argument("--model", type=str, default='lr')
parser.add_argument("--explainer", type=str, default='specific')

parser.add_argument("--save-data", type=str2bool, default=True)
parser.add_argument("--save-path", type=str, default='../Result_iv/Result_Fairness_ML')
parser.add_argument("--skip-exist", type=str2bool, default=False)

args = parser.parse_args()

def subset_data(df, label, k=100, seed=0):
    pos_rate = sum(label == 1.0)/len(label)
    num_pos = int(np.ceil(k*pos_rate))
    num_neg = k - num_pos

    pos_df = shap.utils.sample(df[label == 1.0], num_pos, random_state=seed)
    neg_df = shap.utils.sample(df[label == 0.0], num_neg, random_state=seed)

    return pd.concat([pos_df, neg_df])

def cross_validation(model, x, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    kf.get_n_splits(x)

    acc_list, auc_list = [], []

    for train_index, val_index in kf.split(x):
        x_train = x.iloc[train_index, :]
        x_val = x.iloc[val_index, :]
        y_train = y.iloc[train_index]
        y_val = y.iloc[val_index]

        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        try:
            pred_prob = model.predict_proba(x_val)[:, 1]
        except:
            pred_prob = model._predict_proba_lr(x_val)[:, 1]
        acc_score = accuracy_score(y_val, pred)
        auc_score = roc_auc_score(y_val, pred_prob)

        acc_list.append(acc_score)
        auc_list.append(auc_score)

    print(str(k)+'-Fold cross validation accuracy: ', str(np.mean(acc_list))+'±'+str(np.std(acc_list)))
    print(str(k)+'-Fold cross validation AUC score: ', str(np.mean(auc_list)) + '±' + str(np.std(auc_list)), '\n')


if __name__ == '__main__':
    if args.label_balance and args.test_balance:
        result_path = './stats_ehr_result/ml/' + 'train_test_balanced/' + args.disease
    elif args.label_balance:
        result_path = './stats_ehr_result/ml/' + 'train_balanced/' + args.disease
    elif args.test_balance:
        result_path = './stats_ehr_result/ml/' + 'test_balanced/' + args.disease
    else:
        result_path = './stats_ehr_result/ml/' + 'no_balanced/' + args.disease

    data_df = pd.read_csv(os.path.join(args.data_path, args.file_prefix + args.disease + '.txt'), dtype='float32')
    data_df = data_df[data_df.columns[data_df.columns.map(lambda x: not x.endswith('(Race)'))]]

    if args.icustay:
        result_path = os.path.join(result_path, 'icu_filtered')
        data_df = data_df[data_df['icustay'] == 1.0]
        data_df = data_df.drop('icustay', axis=1)
    # data_df = (data_df - data_df.mean()) / data_df.std()

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.task == 'disease':
        X, y = data_df.iloc[:, 1:-2], data_df.iloc[:, -1]
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    else:
        data_df = data_df[data_df.iloc[:, -1] == 1].iloc[:, :-1]
        X, y = data_df.iloc[:, 1:-1], data_df.iloc[:, -1]
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if args.algorithm_balance:
        scale_pos_weight = (sum(y == 0) / sum(y == 1)) * args.gamma
        class_weight = {0: 1, 1: scale_pos_weight}
    else:
        scale_pos_weight = 1
        class_weight = None

    if args.model.lower() not in ['lr','rf','gbc','svm']:
        print('Not supported model. Please choose from the following four machine learning models: {}'.format(str(['lr','rf','gbc','svm'])))
        sys.exit(1)
    elif args.model.lower() == 'rf':
        classifier = RandomForestClassifier(n_estimators=50, random_state=0, class_weight=class_weight)
    elif args.model.lower() == 'gbc':
        classifier = xgboost.XGBClassifier(n_estimators=50, max_depth=2, random_state=0, class_weight=class_weight)
    elif args.model.lower() == 'svm':
        classifier = LinearSVC(random_state=0, max_iter=2000, class_weight=class_weight)
    else:
        classifier = LogisticRegression(random_state=0, max_iter=2000, class_weight=class_weight)

    print("Dataset Summary:")
    print(y.value_counts(), '\n')
    data_summary = y.value_counts().sort_index()
    data_summary.index = data_summary.index.astype(str) + "_data"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

    print("Train set Summary:")
    print(y_train.value_counts(), '\n')
    train_summary = y_train.value_counts().sort_index()
    train_summary.index = train_summary.index.astype(str) + "_train"

    if args.label_balance:
        resample_index = label_balancer(y_train, method=args.balance_method)
        X_train, y_train = X_train.iloc[resample_index, :], y_train.iloc[resample_index]

        print("Balanced Train set Summary:")
        print(y_train.value_counts(), '\n')
        train_summary_balanced = y_train.value_counts().sort_index()
        train_summary_balanced.index = train_summary_balanced.index.astype(str) + "_train_balance"

    print("Test set Summary:")
    print(y_test.value_counts(), '\n')
    test_summary = y_test.value_counts().sort_index()
    test_summary.index = test_summary.index.astype(str) + "_test"

    if args.test_balance:
        resample_index = label_balancer(y_test, method=args.balance_method)
        X_test_balance, y_test_balance = X_test.iloc[resample_index, :], y_test.iloc[resample_index]

        print("Balanced Test set Summary:")
        print(y_test_balance.value_counts(), '\n')
        test_summary_balanced = y_test_balance.value_counts().sort_index()
        test_summary_balanced.index = test_summary_balanced.index.astype(str) + "_test_balance"

        X_test = X_test_balance
        y_test = y_test_balance

    data_summary = pd.concat([data_summary, train_summary, test_summary])

    metrics_list = list(map(lambda x: x+'_feature_'+str(X_test.shape[1]), ['Acc', 'AUC', 'Recall', 'Precision', 'F1']))\
            +list(map(lambda x: x + '_feature_' + str(args.top), ['Acc', 'AUC', 'Recall', 'Precision', 'F1']))

    cross_validation(classifier, X_train, y_train, 5)

    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    try:
        pred_prob = classifier.predict_proba(X_test)[:, 1]
        def wrapper(x):
            return classifier.predict_proba(x)[:, 1]

    except:
        pred_prob = classifier._predict_proba_lr(X_test)[:, 1]
        def wrapper(x):
            return classifier._predict_proba_lr(x)[:, 1]

    pred_dict = {'label':[], 'pred':[], 'pred_f':[]}
    pred_dict['label'] = y_test
    pred_dict['pred'] = pred_prob

    print('Accuracy on test:', accuracy_score(y_test, pred))
    print('AUC on test:', roc_auc_score(y_test, pred_prob), '\n')

    if not os.path.exists(os.path.join(result_path, 'roc_curve_plots')):
        os.makedirs(os.path.join(result_path, 'roc_curve_plots'))

    #roc_curve_plot(y_test, pred_prob, save_path=os.path.join(result_path, 'roc_curve_plots', args.model+'_roc_curve_raw_data.png'))
    metric_values = [accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prob), recall_score(y_test, pred), precision_score(y_test, pred), f1_score(y_test, pred)]

    if args.explainer == 'kernel':
        X_BG = subset_data(X_train, y_train, 100)
        X_shap = subset_data(X_test, y_test, 100)
        explainer = shap.KernelExplainer(wrapper, X_BG)
        shap_values = explainer(X_shap)
    else:
        X_BG = X_train
        X_shap = X_test
        if args.model.lower() == 'rf':
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(X_shap)
            shap_values = shap_values[:, :, 1]
        elif args.model.lower() == 'gbc':
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(X_shap)
        elif args.model.lower() == 'svm':
            explainer = shap.LinearExplainer(classifier, X_BG)
            shap_values = explainer(X_shap)
        elif args.model.lower() == 'lr':
            explainer = shap.LinearExplainer(classifier, X_BG)
            shap_values = explainer(X_shap)
        else:
            explainer = shap.Explainer(classifier, X_BG)
            shap_values = explainer(X_shap)

    import pickle
    with open(os.path.join(result_path, "shap_explanation_" + args.model + "_" + args.task + ".pkl"), 'wb') as f:
        pickle.dump(shap_values, f)

    fig_path = os.path.join(result_path, "beeswarm_" + args.model + "_" + args.task + ".png")
    if os.path.isfile(fig_path):
        os.remove(fig_path)

    shap.plots.beeswarm(shap_values, max_display=args.top, show=False)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    fig_path = os.path.join(result_path, "shapvalues_" + args.model + "_" + args.task + ".png")
    if os.path.isfile(fig_path):
        os.remove(fig_path)

    shap.plots.bar(shap_values, max_display=args.top, show=False)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    shap_values_abs_mean = shap_values.abs.mean(axis=0)

    # Step 3: Rank Features and Select Top 10
    top_indices = shap_values_abs_mean.values.argsort()[::-1][:args.top]  # Indices of top n features
    rank_indices = shap_values_abs_mean.values.argsort()[::-1]

    shape_value_df = pd.DataFrame({'Feature_names': X.columns[rank_indices].to_list(), 'Shape_values': list(shap_values_abs_mean.values[rank_indices])})
    y_test_r = y_test
    X_train, X_test = X_train.iloc[:, top_indices], X_test.iloc[:, top_indices]

    cross_validation(classifier, X_train, y_train, 5)
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    pred_prob_r = pred_prob
    try:
        pred_prob = classifier.predict_proba(X_test)[:, 1]
    except:
        pred_prob = classifier._predict_proba_lr(X_test)[:, 1]

    pred_dict['pred_f'] = pred_prob
    pred_df = pd.DataFrame(pred_dict)

    print('Accuracy on test with feature filtering:', accuracy_score(y_test, pred))
    print('AUC on test with feature filtering:', roc_auc_score(y_test, pred_prob))

    roc_curves_plot(y_test_r, pred_prob_r, y_test, pred_prob, save_path=os.path.join(result_path, 'roc_curve_plots', args.model+'_roc_curve_filtered_data.png'))
    metric_values += [accuracy_score(y_test, pred), roc_auc_score(y_test, pred_prob), recall_score(y_test, pred),
                     precision_score(y_test, pred), f1_score(y_test, pred)]


    data_summary.to_csv(os.path.join(result_path, "dataset_summary_"+args.task+".csv"))
    pd.DataFrame({'Metrics': metrics_list, 'Values': metric_values}).to_csv(os.path.join(result_path, "result_"+args.model+"_"+args.task+".csv"))
    shape_value_df.to_csv(os.path.join(result_path, "shap_values_"+args.model+"_"+args.task+".csv"))
    pred_df.to_csv(os.path.join(result_path, "pred_"+args.model+"_"+args.task+".csv"))
















