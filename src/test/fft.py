# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:12:27 2021

@author: styra
"""
from cmath import cos, pi, sin

class FFT_impl():
    # _list 是传入的待计算的离散序列，N是序列采样点数，必须是2^n
    def __init__(self, _list=[], N=0):
        self.list = _list  # 初始化数据
        self.N = N
        self.total_m = 0  # 序列的总层数
        self._reverse_list = []  # 位倒序列表
        self.output =  []  # 计算结果存储列表
        self._W = []  # 系数因子列表
        for _ in range(len(self.list)):
            self._reverse_list.append(self.list[self._reverse_pos(_)])
        self.output = self._reverse_list.copy()
        for _ in range(self.N):
            # 提前计算W值，降低算法复杂度
            self._W.append((cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** _)  

    def _reverse_pos(self, num) -> int:  # 得到位倒序后的索引
        out = 0
        bits = 0
        _i = self.N
        data = num
        while (_i != 0):
            _i = _i // 2
            bits += 1
        for i in range(bits - 1):
            out = out << 1
            out |= (data >> i) & 1
        self.total_m = bits - 1
        return out

    def fft(self, _list, N, abs=True) -> list:  # 傅里叶变换处理
        """参数abs=True表示输出结果是否取得绝对值"""
        self.__init__(_list, N)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[__ * 2 ** (self.total_m - m - 1)]
                    self.output[_ * num_each + __] = (temp + temp2)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)
        if abs == True:
            for _ in range(len(self.output)):
                self.output[_] = self.output[_].__abs__()
        return self.output

if __name__ == '__main__':
   input = [0, 1, 2, 3, 4, 5, 6, 7]
   result = FFT_impl().fft(input, 8, False)
   print(result)
