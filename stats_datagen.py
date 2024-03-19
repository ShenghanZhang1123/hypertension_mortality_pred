import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import random
import warnings
from codemappings import icd9to10, get_ccs, convert_drug_to_atc
from arg_tools import str2bool, str2list, check_exist
import sys
import os
from datetime import datetime

# os.chdir('./EHR_Fairness/Mimic')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--mimic-root", type=str, default='../../mimic-iv-2.2')
parser.add_argument("--disease", type=str, default='hypertension')
parser.add_argument("--control", type=str, default='')

parser.add_argument("--tensor-format", type=str, default='temporal')

parser.add_argument("--disease-icd", type=str, default='I10,I11,I12,I13,I14,I15,I16,I1A,I270,I272')
parser.add_argument("--nondisease_icd", type=str, default='')

parser.add_argument("--complication_icd", type=str, default='E08,E09,E10,E11,E12,E13,E14,I21,I22,I49,I50,I63,I64,J12,J13,J14,J15,J16,J17,J18,J44,J45,F01,F02,F03')

parser.add_argument("--new-var", type=str2bool, default=True)
parser.add_argument("--icustay", type=str2bool, default=False)
parser.add_argument("--race-filter", type=str2bool, default=False)

parser.add_argument("--imputation", type=str, default='median')
parser.add_argument("--nan-rate", type=float, default=0.8)

parser.add_argument("--tar-key", type=str, default='subject_id')
parser.add_argument("--icd-key", type=str, default='icd_code')
parser.add_argument("--cpt-key", type=str, default='cpt_cd')
parser.add_argument("--sub-key", type=str, default='subject_id')
parser.add_argument("--item-key", type=str, default='itemid')
parser.add_argument("--value-key", type=str, default='valuenum')
parser.add_argument("--ndc-key", type=str, default='ndc')
parser.add_argument("--drug-key", type=str, default='drug')
parser.add_argument("--gsn-key", type=str, default='gsn')
parser.add_argument("--dose-key", type=str, default='dose_val_rx')
parser.add_argument("--hadm-key", type=str, default='hadm_id')
parser.add_argument("--race-key", type=str, default='race')
parser.add_argument("--gender-key", type=str, default='gender')
parser.add_argument("--age-key", type=str, default='anchor_age')
parser.add_argument("--mortality", type=str, default='hospital_expire_flag')
parser.add_argument("--lan-key", type=str, default='language')
parser.add_argument("--rel-key", type=str, default='religion')
parser.add_argument("--mar-key", type=str, default='marital_status')
parser.add_argument("--admt-key", type=str, default='admission_type')
parser.add_argument("--adml-key", type=str, default='admission_location')
parser.add_argument("--ins-key", type=str, default='insurance')
parser.add_argument("--reg-key", type=str, default='admission_location')
parser.add_argument("--save-data", type=str2bool, default=True)
parser.add_argument("--save-path", type=str, default='../Data_iv')
parser.add_argument("--random_select", type=str2bool, default=True)
parser.add_argument("--subset", type=int, default=None)
# parser.add_argument("--save-format", type=str, default='')

args = parser.parse_args()

def time_diff(intime, outtime):
    # Convert string to datetime objects
    intime_dt = datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")
    outtime_dt = datetime.strptime(outtime, "%Y-%m-%d %H:%M:%S")

    # Calculate the difference in seconds
    time_difference_hours = (outtime_dt - intime_dt).total_seconds()/3600

    return time_difference_hours

if __name__ == '__main__':
    disease = args.disease
    control = args.control

    pat_df = pd.read_csv(os.path.join(args.mimic_root, 'hosp', 'patients.csv.gz'),
                         compression='gzip',
                         header=0,
                         on_bad_lines='skip')
    icu_df = pd.read_csv(os.path.join(args.mimic_root, 'icu', 'icustays.csv.gz'),
                         compression='gzip',
                         header=0,
                         on_bad_lines='skip')
    adm_df = pd.read_csv(os.path.join(args.mimic_root, 'hosp', 'admissions.csv.gz'),
                         compression='gzip',
                         header=0,
                         on_bad_lines='skip')
    diag_df = pd.read_csv(os.path.join(args.mimic_root, 'hosp', 'diagnoses_icd.csv.gz'),
                          compression='gzip',
                          header=0,
                          on_bad_lines='skip')
    diag_dict = pd.read_csv(os.path.join(args.mimic_root, 'hosp', 'd_icd_diagnoses.csv.gz'),
                          compression='gzip',
                          header=0,
                          on_bad_lines='skip')
    lab_df = pd.read_csv(os.path.join(args.mimic_root, 'hosp', 'labevents.csv.gz'),
                         compression='gzip',
                         header=0,
                         on_bad_lines='skip',
                         usecols=[args.sub_key, args.hadm_key, args.item_key, args.value_key])
    lab_dict = pd.read_csv(os.path.join(args.mimic_root, 'hosp', 'd_labitems.csv.gz'),
                           compression='gzip',
                           header=0,
                           on_bad_lines='skip')
    prep_df = pd.read_csv(os.path.join(args.mimic_root, 'hosp', 'prescriptions.csv.gz'),
                          compression='gzip',
                          header=0,
                          on_bad_lines='skip')
    chart_df = pd.read_csv(os.path.join(args.mimic_root, 'icu', 'chartevents.csv.gz'),
                          compression='gzip',
                          header=0,
                          on_bad_lines='skip',
                          usecols=[args.sub_key, args.hadm_key, args.item_key, args.value_key])
    chart_dict = pd.read_csv(os.path.join(args.mimic_root, 'icu', 'd_items.csv.gz'),
                           compression='gzip',
                           header=0,
                           on_bad_lines='skip')

    diag_df[args.icd_key] = diag_df[args.icd_key].str.strip()
    diag_df = diag_df[[args.sub_key, args.hadm_key, args.icd_key]].sort_values(by=[args.sub_key, args.hadm_key])
    # diag_df.drop_duplicates(inplace=True)
    diag_df = diag_df.dropna()

    diag_df[args.icd_key] = icd9to10(list(diag_df[args.icd_key]))
    diag_df = diag_df.dropna()

    Disease_icd_in = np.unique(
        np.asarray([icd for icd in diag_df[args.icd_key] if check_exist(icd, str2list(args.disease_icd))]))
    Disease_icd_comp = np.unique(
        np.asarray([icd for icd in diag_df[args.icd_key] if check_exist(icd, str2list(args.complication_icd))]))
    random.seed(33)
    Disease_icd_ex = np.unique(
        np.asarray([icd for icd in diag_df[args.icd_key] if check_exist(icd, str2list(args.nondisease_icd))]))

    Disease_Patient_IDs_ = diag_df[diag_df[args.icd_key].isin(Disease_icd_in)][args.tar_key].unique()
    Disease_Patient_IDs_ = list(Disease_Patient_IDs_)

    Disease_Patient_EX = diag_df[diag_df[args.icd_key].isin(Disease_icd_ex)][args.tar_key].unique()
    Disease_Patient_EX = set(list(Disease_Patient_EX))
    Disease_Patient_IDs = set(Disease_Patient_IDs_).difference(Disease_Patient_EX)

    if args.subset:
        try:
            Disease_Patient_IDs = random.sample(Disease_Patient_IDs, args.subset)
        except:
            pass

    if args.random_select:
        NonDisease_Patient_IDs = diag_df[~diag_df[args.tar_key].isin(Disease_Patient_IDs)][args.tar_key].unique()
        NonDisease_Patient_IDs = list(NonDisease_Patient_IDs)
        num_nondisease = len(Disease_Patient_IDs)
        NonDisease_Patient_IDs = set(random.sample(NonDisease_Patient_IDs, int(num_nondisease)))
    else:
        NonDisease_Patient_IDs = Disease_Patient_EX
    # NonDisease_Patient_IDs = set(Disease_Patient_IDs_).intersection(Disease_Patient_EX)

    icustay_patient_ids = set(icu_df[args.sub_key].unique().tolist())
    if args.icustay:
        Disease_Patient_IDs = Disease_Patient_IDs.intersection(icustay_patient_ids)
        NonDisease_Patient_IDs = NonDisease_Patient_IDs.intersection(icustay_patient_ids)

    hadm_pos_set = diag_df[diag_df[args.tar_key].isin(Disease_Patient_IDs)][args.hadm_key].unique()
    hadm_neg_set = diag_df[diag_df[args.tar_key].isin(NonDisease_Patient_IDs)][args.hadm_key].unique()
    hadm_set = np.concatenate([hadm_pos_set, hadm_neg_set])

    print("Disease_Patient_IDs", len(Disease_Patient_IDs))
    print("NonDisease_Patient_IDs", len(NonDisease_Patient_IDs))
    print("Intersection", len(set(Disease_Patient_IDs).intersection(NonDisease_Patient_IDs)))
    print("Difference", len(set(Disease_Patient_IDs).difference(NonDisease_Patient_IDs)))

    pat_df = pat_df[[args.sub_key, args.gender_key, args.age_key]].sort_values(by=args.sub_key)
    pat_df.drop_duplicates(subset=[args.sub_key], keep='last', inplace=True)

    adm_df['staylen_seconds'] = adm_df.apply(lambda x: time_diff(x['admittime'], x['dischtime']), axis=1)
    staylen_dict = adm_df.groupby(args.sub_key).staylen_seconds.agg('sum').to_dict()

    adm_df = adm_df[[args.sub_key, args.hadm_key, args.admt_key, args.adml_key, args.ins_key, args.race_key,
                     args.mortality]].sort_values(by=[args.sub_key, args.hadm_key])
    adm_df.drop_duplicates(subset=[args.sub_key, args.hadm_key], keep='first', inplace=True)

    static_df = pat_df.merge(
        adm_df[[args.sub_key, args.admt_key, args.adml_key, args.ins_key, args.race_key, args.mortality]]
        .drop_duplicates(subset=[args.sub_key], keep='first', inplace=False), on=[args.sub_key],
        how='inner').sort_values(by=args.sub_key)

    demo_df = static_df.copy()
    static_list = list(static_df.keys())
    static_list_dic = {}

    for static in static_list[1:]:
        static_list_dic[static] = sorted(list(static_df[static].dropna().unique()))

        def encode(x):
            try:
                return static_list_dic[static].index(x)
            except:
                return -1


        static_df[static] = static_df[static].apply(encode)

    if args.race_filter:
        race_index = set([static_list_dic['race'].index(r) for r in static_list_dic['race']
                          if r.startswith(('AMERICAN INDIAN', 'ASIAN', 'BLACK', 'HISPANIC', 'WHITE'))])

        static_case_df = static_df[static_df[args.tar_key].isin(Disease_Patient_IDs)]
        static_case_df = static_case_df[static_case_df[args.race_key].isin(race_index)]
        Disease_Patient_IDs = set(list(static_case_df[args.tar_key].unique()))

        static_control_df = static_df[static_df[args.tar_key].isin(NonDisease_Patient_IDs)]
        static_control_df = static_control_df[static_control_df[args.race_key].isin(race_index)]
        NonDisease_Patient_IDs = set(list(static_control_df[args.tar_key].unique()))

    cate_var_list = [args.admt_key, args.adml_key, args.ins_key, args.race_key]
    cate_df_list = []

    demo_df['values'] = [1] * len(demo_df)
    demo_df = demo_df[demo_df[args.tar_key].isin(Disease_Patient_IDs.union(NonDisease_Patient_IDs))]
    for var in cate_var_list:
        tem_demo_df = demo_df.pivot_table(index=args.tar_key, columns=var, values='values', aggfunc='first',
                                          fill_value=0.0).sort_values(by=args.tar_key)
        tem_demo_df.columns = tem_demo_df.columns + ' ('+' '.join(list(map(lambda x: x.capitalize(), var.split('_'))))+')'
        cate_df_list.append(tem_demo_df)
    demo_comb_df = pd.concat(cate_df_list, axis=1)

    static_df = static_df.drop(cate_var_list, axis=1)


    diag_df['values'] = [1]*len(diag_df)
    diag_df = (diag_df[diag_df[args.tar_key].isin(Disease_Patient_IDs.union(NonDisease_Patient_IDs)) & diag_df[
        args.hadm_key].isin(hadm_set)]
              .pivot_table(index=args.tar_key, columns=args.icd_key, values='values', aggfunc='first', fill_value=0.0)
              ).sort_values(by=args.tar_key)
    diag_major_df = diag_df[Disease_icd_comp]


    lab_df = lab_df[[args.sub_key, args.hadm_key, args.item_key, args.value_key]]
    lab_df = (lab_df[lab_df[args.tar_key].isin(Disease_Patient_IDs.union(NonDisease_Patient_IDs)) & lab_df[args.hadm_key].isin(hadm_set)]
                   .pivot_table(index=args.tar_key, columns=args.item_key, values=args.value_key, aggfunc='mean')
                   ).sort_values(by=args.tar_key)

    nan_proportions = [sum(lab_df.iloc[:, i].isna()) / len(lab_df) for i in range(len(lab_df.keys()))]
    major_index = [i for i, pro in enumerate(nan_proportions) if pro <= args.nan_rate]
    labevent_major_df = lab_df.iloc[:, major_index]
    # fill_values = {key:labevent_major_df[key].mean() for key in labevent_major_df.keys()}
    # labevent_major_df = labevent_major_df.fillna(fill_values)
    if args.imputation:
        fill_df = labevent_major_df.dropna().agg(args.imputation)
        labevent_major_df = labevent_major_df.fillna(fill_df)
    else:
        labevent_major_df = labevent_major_df.fillna(0.0)

    def filter_float(x):
        try:
            if x != 0:
                return int(x)
        except:
            return None

    def filter_dose(x):
        try:
            return float(x)
        except:
            return None


    prep_df = prep_df[[args.sub_key, args.hadm_key, args.drug_key, args.ndc_key, args.gsn_key, args.dose_key]]
#    drug_dict = dict((drug,ndc) for drug,ndc in zip(prep_df[args.drug_key],prep_df[args.ndc_key]))
#    prep_df[args.ndc_key] = prep_df[args.ndc_key].apply(filter_float)
#    prep_df = prep_df.dropna()
#    prep_df[args.ndc_key] = prep_df[args.ndc_key].astype(np.int64)
    prep_df[args.dose_key] = prep_df[args.dose_key].apply(filter_dose)
    med_df = (prep_df[prep_df[args.tar_key].isin(Disease_Patient_IDs.union(NonDisease_Patient_IDs)) & prep_df[args.hadm_key].isin(hadm_set)]
              .pivot_table(index=args.tar_key, columns=args.drug_key, values=args.dose_key, aggfunc='mean')
              ).sort_values(by=args.tar_key)

    nan_proportions = [sum(med_df.iloc[:, i].isna()) / len(med_df) for i in range(len(med_df.keys()))]
    major_index = [i for i, pro in enumerate(nan_proportions) if pro <= args.nan_rate]
    med_major_df = med_df.iloc[:, major_index]
    # fill_values = {key: med_major_df[key].mean() for key in med_major_df.keys()}
    # med_major_df = med_major_df.fillna(fill_values)
    if args.imputation:
        fill_df = med_major_df.dropna().agg(args.imputation)
        med_major_df = med_major_df.fillna(fill_df)
    else:
        med_major_df = med_major_df.fillna(0.0)

    chart_df = chart_df[[args.sub_key, args.hadm_key, args.item_key, args.value_key]]
    chart_df = (chart_df[chart_df[args.tar_key].isin(Disease_Patient_IDs.union(NonDisease_Patient_IDs)) & chart_df[args.hadm_key].isin(hadm_set)]
                   .pivot_table(index=args.tar_key, columns=args.item_key, values=args.value_key, aggfunc='mean')
                   ).sort_values(by=args.tar_key)

    nan_proportions = [sum(chart_df.iloc[:, i].isna()) / len(chart_df) for i in range(len(chart_df.keys()))]
    major_index = [i for i, pro in enumerate(nan_proportions) if pro <= args.nan_rate]
    chartevent_major_df = chart_df.iloc[:, major_index]
    # fill_values = {key: chartevent_major_df[key].mean() for key in chartevent_major_df.keys()}
    # chartevent_major_df = chartevent_major_df.fillna(fill_values)
    if args.imputation:
        fill_df = chartevent_major_df.dropna().agg(args.imputation)
        chartevent_major_df = chartevent_major_df.fillna(fill_df)
    else:
        chartevent_major_df = chartevent_major_df.fillna(0.0)

    static_df = static_df[static_df[args.tar_key].isin(Disease_Patient_IDs.union(NonDisease_Patient_IDs))]

    #diag_names = diag_dict[diag_dict[args.icd_key].isin(diag_major_df.keys())]
    diag_major_df.columns = diag_major_df.columns + ' (Complication in ICD-10)'

    lab_names = lab_dict[lab_dict[args.item_key].isin(labevent_major_df.keys())]
    labevent_major_df.columns = lab_names['label'].apply(lambda x: x + ' (Lab)')

    chart_names = chart_dict[chart_dict[args.item_key].isin(chartevent_major_df.keys())]
    chartevent_major_df.columns = chart_names['label'].apply(lambda x: x + ' (Lab in ICU stay)')

    med_dict = prep_df[[args.drug_key, args.ndc_key, args.gsn_key]].groupby(args.drug_key).agg({
                    args.ndc_key: set,
                    args.gsn_key: set
                }).reset_index()
    med_names = med_dict[med_dict[args.drug_key].isin(med_major_df.keys())]
    med_major_df.columns = med_major_df.columns.map(lambda x: x + ' (Medication)')

    item_path = './item_names'
    if not os.path.exists(item_path):
        os.makedirs(item_path)

    lab_names.to_csv(os.path.join(item_path, 'labnames.txt'), index=False)
    med_names.to_csv(os.path.join(item_path, 'mednames.txt'), index=False)
    chart_names.to_csv(os.path.join(item_path, 'chartnames.txt'), index=False)

    if args.new_var:
        data_df = diag_major_df.merge(labevent_major_df, on=[args.sub_key], how='outer')\
            .merge(med_major_df, on=[args.sub_key], how='outer')\
            .merge(chartevent_major_df, on=[args.sub_key], how='outer')\
            .merge(demo_comb_df, on=[args.sub_key], how='outer').sort_values(by=args.sub_key)
    else:
        data_df = labevent_major_df.merge(med_major_df, on=[args.sub_key], how='outer') \
            .merge(chartevent_major_df, on=[args.sub_key], how='outer') \
            .merge(demo_comb_df, on=[args.sub_key], how='outer').sort_values(by=args.sub_key)

    data_df = data_df.reset_index()

    if args.new_var:
        data_df['ICU Stay'] = data_df[args.sub_key].apply(lambda x: int(x in icustay_patient_ids))
        data_df['Hospital Stay Length'] = data_df[args.sub_key].apply(lambda x: staylen_dict[x] if x in staylen_dict else 0)

    data_df = data_df.merge(static_df, on=[args.sub_key], how='inner').sort_values(by=args.sub_key)
    if args.imputation:
        fill_df = data_df.dropna().agg(args.imputation)
        data_df = data_df.fillna(fill_df)
    else:
        data_df = data_df.fillna(0.0)
    data_df['label'] = data_df[args.tar_key].apply(lambda x: 1 if x in Disease_Patient_IDs else 0)


    if args.new_var:
        data_df.to_csv(os.path.join(args.save_path, 'new_var', 'complication_chart_lab_drug_'+args.disease+'.txt'),
                       index=False)
    else:
        data_df.to_csv(os.path.join(args.save_path, 'raw', 'chart_lab_drug_' + args.disease + '.txt'),
                       index=False)



