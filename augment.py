# @Time : 2019/03/20 

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from logging import error
from eda import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="原始数据的输入文件目录")
ap.add_argument("--output", required=False, type=str, help="增强数据后的输出文件目录")
ap.add_argument("--num_aug", required=False, type=int, help="每条原始语句增强的语句数")
ap.add_argument("--alpha_sr", required=False, type=float, help="synonyms replacement alpha")
ap.add_argument("--alpha_ri", required=False, type=float, help="random insertion alpha")
ap.add_argument("--alpha_rs", required=False, type=float, help="random swap alpha")
ap.add_argument("--p_rd", required=False, type=float, help="prob of random delete")
ap.add_argument("--alpha", required=False, type=float, help="每条语句中将会被改变的单词数占比")
args = ap.parse_args()

#输出文件
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#每条原始语句增强的语句数
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#每条语句中将会被改变的单词数占比
alpha = 0.1 #default
if args.alpha:
    alpha = args.alpha

#Detailed
alpha_sr = 0.1
if args.alpha_sr:
    alpha_sr = args.alpha_sr
    
alpha_ri = 0.1   
if args.alpha_ri:
    alpha_ri = args.alpha_ri

alpha_rs = 0.1
if args.alpha_rs:
    alpha_rs = args.alpha_rs

p_rd = 0.1
if args.p_rd:
    p_rd = args.p_rd

def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug=9):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()
    length = len(lines)
    region = length // 100
    print(f'corpus length is {length}')

    print("===============Your Set is===============")
    print('''
input_file:{input}\n
output_file: {output}\n
num_of_the_aug_sentences: {num_aug}\n
alpha_of_the_synonym_replacement: {alpha_sr}\n
alpha_of_the_random_insertion: {alpha_ri}\n
alpha_of_the_random_swap: {alpha_rs}\n
prob_of_the_random_delete: {p_rd}\n'''
.format_map(vars(args)))

    print("\n正在使用EDA生成增强语句...")
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')    #使用[:-1]是把\n去掉了
        label = parts[0]
        sentence = parts[1]
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=p_rd, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "\t" + aug_sentence + '\n')

        # add status print
        try:
            j = i+1
            if j%region == 0:
                print(f'{j/region}% completed')
        except error as e:
            print(e)

    writer.close()
    print("已生成增强语句!")
    print(output_file)

if __name__ == "__main__":
    gen_eda(args.input, output, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug=num_aug)