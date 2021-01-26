#%%
import pandas as pd
import numpy as np
import os
import re
import time
from functools import wraps, reduce
from copy import deepcopy
from sklearn.model_selection import train_test_split


#%%
project_dict_second = {
    '01':'有源手术器械',
    '02':'无源手术器械',
    '03':'神经和心血管手术器械',
    '04':'骨科手术器械',
    '05':'耳鼻喉手术器械',
    '06':'医用成像器械',
    '07':'医用诊察监护器械',
    '08':'呼吸麻醉急救器械',
    '09':'物理治疗器械',
    '10':'矫形外科(骨科)手术器械',
    '11':'医疗器械消毒灭菌器械',
    '12':'妇产科用手术器械',
    '13':'计划生育手术器械',
    '14':'注射护理防护器械',
    '16':'眼科器械',
    '17':'口腔科器械',
    '18':'妇产科辅助生殖和避孕器械',
    '19':'物理治疗及康复设备',
    '20':'中医器械',
    '21':'医用诊察和监护器械',
    '22':'医用光学器具仪器及内窥镜设备',
    '23':'医用超声仪器及有关设备',
    '24':'医用激光仪器设备',
    '25':'医用高频仪器设备',
    '26':'物理治疗及康复设备',
    '27':'中医器械',
    '28':'医用成像器械(磁共振)',
    '30':'医用成像器械(X射线)',
    '31':'医用成像器械(成像辅助)',
    '32':'放射治疗器械',
    '33':'医用核素设备',
    '40':'临床检验器械',
    '41':'临床检验器械',
    '45':'输血透析和体外循环器械',
    '46':'认知言语试听障碍康复设备',
    '50':'??????',
    '54':'手术室急救室诊疗室设备及器具',
    '55':'口腔科器械',
    '56':'病房护理设备及器具',
    '57':'医疗器械消毒灭菌器械',
    '58':'医用冷疗低温冷藏设备及器具',
    '63':'口腔科器械',
    '64':'注射护理防护器械',
    '65':'医用缝合材料及粘合剂',
    '66':'医用高分子材料及制品',
    '70':'医用软件',
    '77':'无源手术器械'
}

project_dict_third = {
    '01':'有源手术器械',
    '02':'无源手术器械',
    '03':'神经和心血管手术器械',
    '04':'骨科手术器械',
    '05':'放射治疗器械',
    '06':'医用成像器械',
    '07':'医用侦察和监护器械',
    '08':'呼吸麻醉急救器械',
    '09':'物理治疗器械',
    '10':'输血透析体外循环器械',
    '11':'医疗器械消毒灭菌器械',
    '12':'有源植入器械',
    '13':'无源植入器械',
    '14':'注射护理防护器械',
    '15':'注射护理防护器械',
    '16':'眼科器械',
    '17':'口腔科器械',
    '18':'妇产科辅助生殖和避孕器械',
    '19':'物理治疗及康复设备',
    '20':'中医器械',
    '21':'医用诊察和监护器械',
    '22':'医用光学器具仪器及内窥镜设备',
    '23':'医用超声仪器及有关设备',
    '24':'医用激光仪器设备',
    '25':'医用高频仪器设备',
    '26':'物理治疗及康复设备',
    '27':'中医器械',
    '28':'医用成像器械(磁共振)',
    '30':'医用成像器械(X射线)',
    '31':'医用成像器械(成像辅助)',
    '32':'放射治疗器械',
    '33':'医用核素设备',
    '40':'临床检验器械',
    '41':'临床检验器械',
    '45':'输血透析和体外循环器械',
    '46':'无源植入器械',
    '54':'手术室急救室诊疗室设备及器具',
    '58':'物理治疗器械',
    '63':'口腔科器械',
    '64':'注射护理防护器械',
    '65':'医用缝合材料及粘合剂',
    '66':'注输护理和防护器械',
    '70':'医用软件',
    '77':'神经和心血管手术器械'
}


#%%
def benchmark():
    def middle(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            result = f(*args, **kwargs)
            t_end = time.time()
            print(f'{f.__name__} takes {t_end-t_start}')
            return result
        return wrapper
    return middle


@benchmark()
def rm_element(content: list, condition: str) -> list:
    result = deepcopy(content)
    for i in result[::-1]:
        if re.match(re.compile(condition), i) is not None:
            pass
        else:
            result.remove(i)
    return result


@benchmark()
def data_preprocess(folder_name: str,
                    project_dict_second: dict,
                    project_dict_third: dict) -> pd.DataFrame:

    # get the list of data files
    file_list = os.listdir(folder_name)

    # rm the unrelated file
    df_list = rm_element(file_list, '[0-9a-zA-Z\_]+\.csv')

    # loop the list, transfer it into pd.DataFrame and concat
    for i, j in enumerate(df_list):
        df_list[i] = pd.read_csv(folder_name+'/'+j)

    df = pd.concat(df_list)

    # select the data from 2010-2019
    df = df[df['注册证编号'].str.match(r'[\u4e00-\u9fa5]{1,5}201[1-9]{1}[23]{1}[0-7]{1}[0-9]{1}[0-9]{4}')]

    # select the target columns
    df = df.loc[:, ['注册证编号','产品名称','结构及组成/主要组成成分', '适用范围/预期用途']]

    # replace the class row
    df['注册证编号'] = df['注册证编号'].str.extract(r'(201[1-9]{1}[23]{1}[0-7]{1}[0-9]{1}[0-9]{4})')
    df['注册证编号'] = df['注册证编号'].str[4:7]

    # fill NaN
    df = df.fillna(value='')

    # concat text columns
    def column_add(c1: pd.Series, c2: pd.Series) -> pd.Series:
        return c1.str.cat(c2, sep=' ')

    df['合并数据'] = reduce(column_add, [df[column] for column in df.columns][1:])

    df = df[['注册证编号','合并数据']]

    # reset Index
    df.reset_index(drop=True)

    # rename dataframe
    df = df.rename(columns={'注册证编号':'label', '合并数据':'text'})

    # drop the \r\n\t in the str in the text col
    df['text'] = df['text'].str.replace('(\\t|\\n|\\r)', '', regex=True)

    # drop 215 cause manual check for conflict
    df = df[(df['label'] != '215') & (df['label'] != '234')]

    # drop duplicates
    df = df.drop_duplicates()

    # drop the label which just got 6 or less examples
    df = df.groupby('label').filter(lambda x: len(x) > 13)

    # transfer label into str
    def project(x: int, project_dict_second: dict, project_dict_third:dict) -> str:
        # type transform
        try:
            data = str(x)
        except:
            pass

        data = str(x)

        # administration category
        if data[0] == '2':
            result = '第二类'
            result = result + project_dict_second[data[1:]]   
        else:
            result = '第三类'
            result = result + project_dict_third[data[1:]]

        return result

    df['label'] = df['label'].apply(project, args=(project_dict_second, project_dict_third))

    # split them in to train, val, test
    modelling, val = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    train, test = train_test_split(modelling, test_size=0.11, stratify=modelling['label'], random_state=42)

    return train, val, test


#%%
if __name__ == '__main__':
    train, val, test = data_preprocess('data_folder', project_dict_second, project_dict_third)

    print(len(train))
    print(len(val))
    print(len(test))

    train.to_csv('cleaned_data/training_data.txt', index=False, header=False, sep='\t', encoding='utf-8')
    val.to_csv('cleaned_data/val_data.txt', index=False, header=False, sep='\t', encoding='utf-8')
    test.to_csv('cleaned_data/test_data.txt',index=False, header=False, sep='\t', encoding='utf-8')

    print(train.groupby(['label']).count())

    print(val.groupby(['label']).count())

    print(test.groupby(['label']).count())

    print(train.head())

    # when filter is > 11, label = 63
    # when filter is > 6, label = 67