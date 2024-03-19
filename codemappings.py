import csv
import pandas as pd
from tqdm import tqdm
import re

def _clnrw(row):
    return [x.replace('"',"").replace("'","").strip() for x in row]

def read_ccs(fn):
    icd2ccs = {}
    with open(fn, "r") as fp:
        reader = csv.reader(fp, delimiter=",")
        header = next(reader)
        for row in reader:
            row = _clnrw(row)
            icd2ccs[row[0]] = {"ccs": row[1],
                            "ccs_desc": row[3],
                            "ccs_lv1": row[4],
                            "ccs_lv1_desc": row[5],
                            "ccs_lv2": row[6],
                            "ccs_lv2_desc": row[7]}
    return icd2ccs

def icd9to10(icd9_list):
    data_table = {}
    try:
        f1 = open('../../ATC_ICD_CCS/icd9to10dictionary.txt')
        for line in f1:
            nine = str.strip(line.split('|')[0])
            ten = str.strip(line.split('|')[1])
            data_table[nine.replace('.','').upper()] = ten.replace('.','').upper()
    except FileNotFoundError:
        print("Missing dependency: icd9to10dictionary.txt")
    icd_list = []
    for icd9 in tqdm(icd9_list):
        try:
            icd_list.append(data_table[icd9])
        except:
            #print('KeyError: ', icd9)
            if icd9 in data_table.values():
                icd_list.append(icd9)
            else:
                icd_list.append(None)
    return icd_list


def get_ccs(x_lst):
    fn = "../../ATC_ICD_CCS/ccs_dx_icd10cm_2019_1.csv"
    x2ccs = read_ccs(fn)

    x_lst = [x.strip().upper().replace(".", "") for x in x_lst]
    ccs_lst = []
    out_default = {"ccs": None,
                   "ccs_desc": "na"}
    for x in tqdm(x_lst):
        if x not in x2ccs:
            ccs_lst.append(out_default['ccs'])
        else:
            ccs_lst.append(x2ccs[x]['ccs'])

    out = ccs_lst
    return out

def convert_drug_to_atc(drug_name):
    # Load the ATC/DDD Index
    atc_index = pd.read_csv('../../ATC_ICD_CCS/ATC-DDD.csv')  # Replace with the path to your ATC/DDD Index file

    atc_list = []
    for drug in tqdm(drug_name):
        # Filter the Index for the given drug name
        try:
            reg = re.compile('\W')
            name = max(reg.split(drug), key=len, default='')
            filtered_index = atc_index[atc_index['atc_name'].str.contains(name, case=False)]
            # Extract the ATC code
            atc_code = [code for code in list(filtered_index['atc_code']) if len(code) >= 5][0]
        except:
            #print('ATC code not found')
            atc_code = None

        atc_list.append(atc_code)

    return atc_list

'''
def convert_drug_to_atc(drug_name):
    # Load the ATC/DDD Index
    atc_index = pd.read_csv('./ATC_ICD_CCS/ATC-DDD.csv')  # Replace with the path to your ATC/DDD Index file
    data_table = {}
    for atc, drug in zip(list(atc_index['atc_code']), list(atc_index['atc_name'])):
        data_table[str(drug)] = str(atc)

    atc_list = []
    for drug in drug_name:
        # Filter the Index for the given drug name
        try:
            atc_code = data_table[drug]
        except:
            #print('ATC code not found')
            atc_code = None

        atc_list.append(atc_code)
    return atc_list
'''

