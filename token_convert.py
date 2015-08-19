#encoding=utf-8
zh_dir = 'zh-en/IWSLT.zh'
en_dir = 'zh-en/IWSLT.en'

en = open(en_dir,'r')
zh = open(zh_dir,'r')

def number_file(dir):
	with open(dir) as f:
		return sum(1 for _ in f)
		
len_p = number_file(en_dir)

corpus = []
for x in range(0, len_p):
	en_words=[]
	zh_words=[]
	en_words=en.readline().split(" ")
	zh_words=zh.readline().split(" ")
	corpus.append([en_words,zh_words])

###############
print '#####Writing Corpus into a file####'
import os
directory = 'data/'
filename = 'max_token'
if not os.path.exists(directory):
    os.makedirs(directory)
file = open(directory+filename, "w")

file.write(str(corpus))

file.close()
print '#####Successfully Wrote into '+filename+'######'
###############	