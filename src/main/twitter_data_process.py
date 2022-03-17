# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:51:24 2022

@author: styra
"""

dic = {
    'non-rumor': 0,   # Non-rumor   NR
    'false': 1,   # false rumor    FR
    'unverified': 2,  # unverified tweet  UR
    'true': 3,    # debunk rumor  TR
}

root_path = 'data/dataset/twitter16/'
file_name = 'twitter16'
real_content, fake_content = [], []
with open(root_path + file_name +".train", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        if label == 'non-rumor' or label == 'true' :
            real_content.append(content)
        elif label == 'false':
            fake_content.append(content)

with open(root_path + file_name +".dev", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        if label == 'non-rumor' or label == 'true' :
            real_content.append(content)
        elif label == 'false':
            fake_content.append(content)

with open(root_path + file_name +".test", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        if label == 'non-rumor' or label == 'true' :
            real_content.append(content)
        elif label == 'false':
            fake_content.append(content)

rW = open(root_path + 'real.txt', 'a', encoding='UTF-8')
rW.write('\n'.join(real_content))
rW.close()

fW = open(root_path + 'fake.txt', 'a', encoding='UTF-8')
fW.write('\n'.join(fake_content))
fW.close()

