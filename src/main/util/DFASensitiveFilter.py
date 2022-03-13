# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 20:01:01 2022

@author: styra
"""
class DFAFilter(object):
	"""DFA过滤算法"""
	def __init__(self):
		super(DFAFilter, self).__init__()
		self.keyword_chains = {}
		self.delimit = '\x00'

	# 读取解析敏感词
	def parseSensitiveWords(self, path):
		ropen = open(path, 'r', encoding='UTF-8')
		text = ropen.read()
		keyWordList = text.split(',')
		for keyword in keyWordList:
			self.addSensitiveWords(str(keyword).strip())

	# 生成敏感词树
	def addSensitiveWords(self, keyword):
		keyword = keyword.lower()
		chars = keyword.strip()
		if not chars:
			return
		level = self.keyword_chains
		for i in range(len(chars)):
			if chars[i] in level:
				level = level[chars[i]]
			else:
				if not isinstance(level, dict):
					break
				for j in range(i, len(chars)):
					level[chars[j]] = {}

					last_level, last_char = level, chars[j]

					level = level[chars[j]]
				last_level[last_char] = {self.delimit: 0}
				break
			if i == len(chars) - 1:
				level[self.delimit] = 0

	# 过滤敏感词
	def filterSensitiveWords(self, message, repl="*"):
		message = message.lower()
		ret = []
		start = 0
		while start < len(message):
			level = self.keyword_chains
			step_ins = 0
			message_chars = message[start:]
			for char in message_chars:
				if char in level:
					step_ins += 1
					if self.delimit not in level[char]:
						level = level[char]
					else:
						ret.append(repl * step_ins)
						start += step_ins - 1
						break
				else:
					ret.append(message[start])
					break
			start += 1

		return ''.join(ret)
    
def removeSensitiveWords(text) :
    gfw = DFAFilter()
    gfw.parseSensitiveWords('shieldwords.txt')
    return gfw.filterSensitiveWords(text)


if __name__ == "__main__":
	text = "小明骂小王是个王八蛋，小王骂小明是个王八羔子！"
	print(removeSensitiveWords(text))
