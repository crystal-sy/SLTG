# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:12:27 2021

@author: styra
"""

from operator import itemgetter

# 树节点类构建
class TreeNode(object):
  def __init__(self, data):
    self.val = data[0]
    self.priority = data[1]
    self.leftChild = None
    self.rightChild = None
    self.code = ""
# 创建树节点队列函数
def creatnodeQ(codes):
  q = []
  for code in codes:
    q.append(TreeNode(code))
  return q
# 为队列添加节点元素，并保证优先度从大到小排列
def addQ(queue, nodeNew):
  if len(queue) == 0:
    return [nodeNew]
  for i in range(len(queue)):
    if queue[i].priority >= nodeNew.priority:
      return queue[:i] + [nodeNew] + queue[i:]
  return queue + [nodeNew]
# 节点队列类定义
class nodeQeuen(object):
 
  def __init__(self, code):
    self.que = creatnodeQ(code)
    self.size = len(self.que)
 
  def addNode(self,node):
    self.que = addQ(self.que, node)
    self.size += 1
 
  def popNode(self):
    self.size -= 1
    return self.que.pop(0)

# 创建哈夫曼树
def creatHuffmanTree(nodeQ):
  while nodeQ.size != 1:
    node1 = nodeQ.popNode()
    node2 = nodeQ.popNode()
    r = TreeNode([None, node1.priority+node2.priority])
    r.leftChild = node1
    r.rightChild = node2
    nodeQ.addNode(r)
  return nodeQ.popNode()
 
codeDic1 = {}
# 由哈夫曼树得到哈夫曼编码表
def HuffmanCodeDic(head, x):
  global codeDic, codeList
  if head:
    HuffmanCodeDic(head.leftChild, x+'1')
    head.code += x
    if head.val:
      codeDic1[head.val] = head.code
    HuffmanCodeDic(head.rightChild, x+'0')
    
def codeLength(codes):
  l = []
  for code in codes:
   l.append(len(code[1]))
  return l

def avlCodeLength(lengths):
  l = 0
  for length in lengths:
      l += length
  return round(l / len(lengths), 2)


d = {'X2': 0.25, 'X1': 0.25, 'X3': 0.20, 'X4': 0.15, 'X5': 0.10, 'X6': 0.05}
d = sorted(d.items(),key=lambda x:x[1])
code = {}
for num in d:
    code[num[0]] = int(num[1] * 100)
codes = sorted(code.items(), key=itemgetter(1))
t = nodeQeuen(codes)
tree = creatHuffmanTree(t)
HuffmanCodeDic(tree, '')
codeDic1 = sorted(codeDic1.items(),key=lambda x:x[0])
print("码树：")
print(codeDic1)
codeLength = codeLength(codeDic1)
print("码长：")
print(codeLength)
avlCodeLength = avlCodeLength(codeLength)
print("平均码长：")
print(avlCodeLength)
