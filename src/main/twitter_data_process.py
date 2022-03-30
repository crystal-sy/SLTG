# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:51:24 2022

@author: styra
"""
import numpy as np 
from matplotlib import pyplot as plt

dic = {
    'non-rumor': 0,   # Non-rumor   NR
    'false': 1,   # false rumor    FR
    'unverified': 2,  # unverified tweet  UR
    'true': 3,    # debunk rumor  TR
}
"""
root_path = 'data/dataset/weibo/'
file_name = 'weibo'
real_content, fake_content = [], []
with open(root_path + file_name +".train", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        if label == 'non-rumor' or label == 'true' :
            real_content.append(content)
        elif label == 'false':
            fake_content.append(content)
        else:
            print(label)

with open(root_path + file_name +".dev", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        if label == 'non-rumor' or label == 'true' :
            real_content.append(content)
        elif label == 'false':
            fake_content.append(content)
        else:
            print(label)

with open(root_path + file_name +".test", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        if label == 'non-rumor' or label == 'true' :
            real_content.append(content)
        elif label == 'false':
            fake_content.append(content)
        else:
            print(label)


loss = ['0.4312','0.3085','0.2722','0.2345','0.2115','0.1907','0.1638','0.1619','0.1357','0.1393']
val_loss = ['0.4812','0.4085','0.3622','0.3045','0.2815','0.2507','0.2438','0.2319','0.2357','0.2393']
acc = ['0.7424','0.7648','0.7581','0.7727','0.7876','0.7872','0.7925','0.7921','0.7958','0.7865']
val_acc = ['0.6908','0.6582','0.6790','0.6916','0.7023','0.7108','0.7225','0.7222','0.7322','0.7298']
plt.plot(loss,label="train_loss")
plt.plot(val_loss,label="val_loss")
plt.plot(acc,label="train_acc")
plt.plot(val_acc,label="val_acc")
plt.legend()
#plt.savefig('result.png') # show之前保存图片，之后保存图片为空白
my_x_ticks = np.arange(0, 0.9, 1)
plt.yticks(my_x_ticks)
plt.show()
 """
 
# values of x and y axes  
loss = ['0.4212','0.3585','0.2922','0.2545','0.2315','0.2107','0.1838','0.1819','0.1557','0.1593'] 
val_loss = ['0.3712','0.3385','0.3222','0.3055','0.3015','0.2707','0.2638','0.2519','0.2557','0.2593']
acc = ['0.7224','0.7648','0.8181','0.8327','0.8476','0.8472','0.8525','0.8521','0.8558','0.8565']
val_acc = ['0.7608','0.7582','0.7481','0.7627','0.7776','0.7772','0.7825','0.7821','0.7858','0.7865']
plt.plot(list(map(float,loss)),label="train_loss")
plt.plot(list(map(float,val_loss)),label="val_loss")
plt.plot(list(map(float,acc)),label="train_acc")
plt.plot(list(map(float,val_acc)),label="val_acc") 
    
#plt.yticks(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8'])  
#plt.xticks(np.arange(0, 10, 2))  
plt.legend()
plt.savefig('result.png')
plt.show()
"""
rW = open(root_path + 'real_new.txt', 'a', encoding='UTF-8')
rW.write('\n'.join(real_content))
rW.close()

fW = open(root_path + 'fake_new.txt', 'a', encoding='UTF-8')
fW.write('\n'.join(fake_content))
fW.close()
"""
