#encoding=utf-8
token_dir = 'data/max_token'
print '########Reading Corpus From file "'+token_dir+'"###########\n'

f = open(token_dir,'r')

str_corpus = f.readline()

import ast

corpus = ast.literal_eval(str_corpus)

del str_corpus

print '########Finished Reading Corpus from a file#######'
###
print '########Starting shuffle corpus#######'
##
import random

tmp = []
for node in corpus:
	tmp.append(node[1])#eng
random.shuffle(tmp)
##
len_corpus = len(corpus)
for index in range(0, len_corpus):
    corpus[index][1] = tmp[index]
del tmp
print '########End of shuffle corpus#######'
############
print '#####Writing Corpus into a file####'
import os
directory = 'data/'
filename = 'max_token_wrong'
if not os.path.exists(directory):
    os.makedirs(directory)
file = open(directory+filename, "w")

file.write(str(corpus))

file.close()
print '#####Successfully Wrote into '+filename+'######'