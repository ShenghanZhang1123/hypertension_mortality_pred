from arg_tools import str2bool, str2list
import argparse
import warnings
import pandas as pd
import os
#os.chdir('./EHR_Fairness/Mimic')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", type=str, default='../Data_iv/new_var')
parser.add_argument("--disease", type=str, default='hypertension')
parser.add_argument("--control", type=str, default='')
parser.add_argument("--icd-format", type=str, default='CCS')
parser.add_argument("--ATC-level", type=int, default=4)
parser.add_argument("--task", type=str, default='mortality')
parser.add_argument("--mimic-iv", type=str2bool, default=True)

args = parser.parse_args()

data_frame = {'Characteristic (Hypertension)': ['Count','Anchor age','  Mean (SD)','  Median (IQR)','Length of stay','  Mean (SD)','  Median (IQR)','Sex','  Male','  Female','Race',
'  American Indian','  Asian','  Black','  Hispanic','  White','  Others','ICU stay','  Yes',
'  No','Complication','  Diabetes Mellitus','  Heart Failure','  Arrhythmias','  Myocardial Infarction','  Chronic Obstructive Pulmonary Disease (COPD)',
'  Asthma','  Pneumonia','  Stroke','  Dementia'], 'Mortality': [], 'Non-mortality': [], 'Total': []}

DM_list = str2list('E08,E09,E10,E11,E12,E13,E14')
HF_list = str2list('I50')
Ar_list = str2list('I49')
MI_list = str2list('I21,I22')
COPD_list = str2list('J44')
As_list = str2list('J45')
Pn_list = str2list('J12,J13,J14,J15,J16,J17,J18')
St_list = str2list('I63,I64')
De_list = str2list('F01,F02,F03')

def check_num(data_df, name_list):
    name_list = tuple(name_list)
    num = data_df[data_df.columns[data_df.columns.map(lambda x: x.startswith(name_list))]].sum().sum()
    return int(num)

if __name__ == '__main__':
    data_df = pd.read_csv(os.path.join(args.data_path, 'complication_chart_lab_drug_' + args.disease + '.txt'), dtype='float32')
    if args.task == 'disease':
        X, y = data_df.iloc[:, 1:-2], data_df.iloc[:, -1]
    else:
        data_df = data_df[data_df.iloc[:, -1] == 1].iloc[:, :-1]
        X, y = data_df.iloc[:, 1:-1], data_df.iloc[:, -1]

    pos_df = X[y == 1]
    neg_df = X[y == 0]

    data_frame['Mortality'].append('n='+str(len(pos_df)))
    data_frame['Non-mortality'].append('n='+str(len(neg_df)))
    data_frame['Total'].append('n=' + str(len(X)))

    data_frame['Mortality'].append('')
    data_frame['Non-mortality'].append('')
    data_frame['Total'].append('')

    data_frame['Mortality'].append(
        str(pos_df['anchor_age'].mean().round(2)) + ' (' + str(round(pos_df['anchor_age'].std(), 2)) + ')')
    data_frame['Non-mortality'].append(
        str(neg_df['anchor_age'].mean().round(2)) + ' (' + str(round(neg_df['anchor_age'].std(), 2)) + ')')
    data_frame['Total'].append(
        str(X['anchor_age'].mean().round(2)) + ' (' + str(round(X['anchor_age'].std(), 2)) + ')')

    data_frame['Mortality'].append(
        str(pos_df['anchor_age'].median()) + ' (' + str(pos_df['anchor_age'].quantile(0.25)) + '-' + str(pos_df['anchor_age'].quantile(0.75)) + ')')
    data_frame['Non-mortality'].append(
        str(neg_df['anchor_age'].median()) + ' (' + str(neg_df['anchor_age'].quantile(0.25)) + '-' + str(neg_df['anchor_age'].quantile(0.75)) + ')')
    data_frame['Total'].append(
        str(X['anchor_age'].median()) + ' (' + str(X['anchor_age'].quantile(0.25)) + '-' + str(X['anchor_age'].quantile(0.75)) + ')')

    data_frame['Mortality'].append('')
    data_frame['Non-mortality'].append('')
    data_frame['Total'].append('')

    data_frame['Mortality'].append(
        str(pos_df['staylen'].mean().round(2)) + ' (' + str(round(pos_df['staylen'].std(), 2)) + ')')
    data_frame['Non-mortality'].append(
        str(neg_df['staylen'].mean().round(2)) + ' (' + str(round(neg_df['staylen'].std(), 2)) + ')')
    data_frame['Total'].append(
        str(X['staylen'].mean().round(2)) + ' (' + str(round(X['staylen'].std(), 2)) + ')')

    data_frame['Mortality'].append(
        str(round(pos_df['staylen'].median(), 2)) + ' (' + str(round(pos_df['staylen'].quantile(0.25), 2)) + '-' + str(round(pos_df['staylen'].quantile(0.75), 2)) + ')')
    data_frame['Non-mortality'].append(
        str(round(neg_df['staylen'].median(), 2)) + ' (' + str(round(neg_df['staylen'].quantile(0.25), 2)) + '-' + str(round(neg_df['staylen'].quantile(0.75), 2)) + ')')
    data_frame['Total'].append(
        str(round(X['staylen'].median(), 2)) + ' (' + str(round(X['staylen'].quantile(0.25), 2)) + '-' + str(
            round(X['staylen'].quantile(0.75), 2)) + ')')

    data_frame['Mortality'].append('')
    data_frame['Non-mortality'].append('')
    data_frame['Total'].append('')

    data_frame['Mortality'].append(
        str(pos_df['gender'].to_list().count(1.0)) + ' (' + str(round((pos_df['gender'].to_list().count(1.0)/len(pos_df))*100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(neg_df['gender'].to_list().count(1.0)) + ' (' + str(round((neg_df['gender'].to_list().count(1.0)/len(neg_df))*100, 1)) + '%)')
    data_frame['Total'].append(
        str(X['gender'].to_list().count(1.0)) + ' (' + str(
            round((X['gender'].to_list().count(1.0) / len(X)) * 100, 1)) + '%)')

    data_frame['Mortality'].append(
        str(pos_df['gender'].to_list().count(0.0)) + ' (' + str(round((pos_df['gender'].to_list().count(0.0)/len(pos_df))*100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(neg_df['gender'].to_list().count(0.0)) + ' (' + str(round((neg_df['gender'].to_list().count(0.0)/len(neg_df))*100, 1)) + '%)')
    data_frame['Total'].append(
        str(X['gender'].to_list().count(0.0)) + ' (' + str(
            round((X['gender'].to_list().count(0.0) / len(X)) * 100, 1)) + '%)')

    data_frame['Mortality'].append('')
    data_frame['Non-mortality'].append('')
    data_frame['Total'].append('')

    count_pos = check_num(pos_df, ['AMERICAN INDIAN'])
    count_neg = check_num(neg_df, ['AMERICAN INDIAN'])
    count_x = check_num(X, ['AMERICAN INDIAN'])

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos/len(pos_df))*100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg/len(neg_df))*100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, ['ASIAN'])
    count_neg = check_num(neg_df, ['ASIAN'])
    count_x = check_num(X, ['ASIAN'])

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, ['BLACK'])
    count_neg = check_num(neg_df, ['BLACK'])
    count_x = check_num(X, ['BLACK'])

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, ['HISPANIC'])
    count_neg = check_num(neg_df, ['HISPANIC'])
    count_x = check_num(X, ['HISPANIC'])

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, ['WHITE'])
    count_neg = check_num(neg_df, ['WHITE'])
    count_x = check_num(X, ['WHITE'])

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = len(pos_df) - check_num(pos_df, ['AMERICAN INDIAN', 'ASIAN', 'BLACK', 'HISPANIC', 'WHITE'])
    count_neg = len(neg_df) - check_num(neg_df, ['AMERICAN INDIAN', 'ASIAN', 'BLACK', 'HISPANIC', 'WHITE'])
    count_x = len(X) - check_num(X, ['AMERICAN INDIAN', 'ASIAN', 'BLACK', 'HISPANIC', 'WHITE'])

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    data_frame['Mortality'].append('')
    data_frame['Non-mortality'].append('')
    data_frame['Total'].append('')

    count_pos = int(sum(pos_df['icustay'] == 1.0))
    count_neg = int(sum(neg_df['icustay'] == 1.0))
    count_x = int(sum(X['icustay'] == 1.0))

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = int(sum(pos_df['icustay'] == 0.0))
    count_neg = int(sum(neg_df['icustay'] == 0.0))
    count_x = int(sum(X['icustay'] == 0.0))

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    data_frame['Mortality'].append('')
    data_frame['Non-mortality'].append('')
    data_frame['Total'].append('')

    count_pos = check_num(pos_df, DM_list)
    count_neg = check_num(neg_df, DM_list)
    count_x = check_num(X, DM_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, HF_list)
    count_neg = check_num(neg_df, HF_list)
    count_x = check_num(X, HF_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, Ar_list)
    count_neg = check_num(neg_df, Ar_list)
    count_x = check_num(X, Ar_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, MI_list)
    count_neg = check_num(neg_df, MI_list)
    count_x = check_num(X, MI_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, COPD_list)
    count_neg = check_num(neg_df, COPD_list)
    count_x = check_num(X, COPD_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, As_list)
    count_neg = check_num(neg_df, As_list)
    count_x = check_num(X, As_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, Pn_list)
    count_neg = check_num(neg_df, Pn_list)
    count_x = check_num(X, Pn_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, St_list)
    count_neg = check_num(neg_df, St_list)
    count_x = check_num(X, St_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    count_pos = check_num(pos_df, De_list)
    count_neg = check_num(neg_df, De_list)
    count_x = check_num(X, De_list)

    data_frame['Mortality'].append(
        str(count_pos) + ' (' + str(round((count_pos / len(pos_df)) * 100, 1)) + '%)')
    data_frame['Non-mortality'].append(
        str(count_neg) + ' (' + str(round((count_neg / len(neg_df)) * 100, 1)) + '%)')
    data_frame['Total'].append(
        str(count_x) + ' (' + str(round((count_x / len(X)) * 100, 1)) + '%)')

    sum_df = pd.DataFrame(data_frame)
    sum_df.to_csv(os.path.join(args.data_path, 'dataset_feature_summary.csv'))






