import argparse
import ast

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(x):
    stripped_str = x.strip('[]')
    # Split the string by comma
    result_list = stripped_str.split(',')
    try:
        result_list.remove('')
    except:
        pass
    return result_list

def check_exist(icd, list_prefix):
    flag = False
    for prefix in list_prefix:
        if str(icd).startswith(str(prefix)):
            flag = True
    return flag
