# encoding=utf-8

import nltk
nltk.usage(nltk.classify.ClassifierI)
###

import time
import sys
#zh_dir = 'zh-en/'+args['zh_dir']
#en_dir = 'zh-en/'+args['en_dir']
#zh_dir = 'zh-en/IWSLT.zh'
#en_dir = 'zh-en/IWSLT.en'
#zh_dir = 'zh-en/1m_uni.tok.zh'
#en_dir = 'zh-en/1m_uni.tok.en'
#zh_dir = 'zh-en/test.zh'
#en_dir = 'zh-en/test.en'


def read_corpus(en_dir, zh_dir, tag):
	print '########Reading Corpus_%5s' % tag + ' From file "' + zh_dir + '  ' + en_dir + '"###########'
	en = open(en_dir, 'r')
	zh = open(zh_dir, 'r')

	def number_file(dir):
		with open(dir) as f:
			return sum(1 for _ in f)

	len_p = number_file(en_dir)

	corpus = []
	for x in range(0, len_p):
		en_words = []
		zh_words = []
		en_words = en.readline().split(" ")
		zh_words = zh.readline().split(" ")
		corpus.append([en_words, zh_words])
	return corpus





def shuffle_corpus(corpus):
	print '########Starting shuffle corpus#######'
	##
	import random

	tmp = []
	for node in corpus:
		tmp.append(node[1])  # eng
	random.shuffle(tmp)
	##
	len_corpus = len(corpus)
	for index in range(0, len_corpus):
		corpus[index][1] = tmp[index]
	del tmp
	print '########End of shuffle corpus#######'
	return corpus



from lex.readLexicalModelIntoDict_set import readLex_set

def prepare_lexical(my_dict,lex_table):
	lex_fname = lex_table
	#lexical_file = 'zh-en/'+lex_fname  # chinese_index
	lexical_file = lex_fname
	# lexical_file = 'zh-en/test_f'#chinese_index
	# lexical_file = 'zh-en/lex.e2f'+'_f'#chinese_index
	print '###Preparing LexicalModelDict ' + lexical_file + '###'
	order = 'e2f'
	my_dict = readLex_set(lexical_file,order)
	
	return my_dict



def count_search_dict(word):
	try:
		return len(my_dict[word])
		# return 0
	except KeyError:
		return 0

def count_match_dict(source, target):
	try:
		candidate_list = my_dict[source]
		if target in candidate_list:
			return 1
		else:
			return 0
			# print '-target- ' + target +' -word- ' + word + ' -source- ' + source
		#
	except KeyError:
		return 0


def gender_features(en, zh):
	features = {}
	
	# num of words
	features["num_eng"] = len(en)
	features["num_zh"] = len(zh)
	##freq_word_features##
	'''
	doc_words_en = set(en)
	doc_words_zh = set(zh)
	#run = 0
	
	w_list = []
	for word in word_features_en:
		if (word in doc_words_en): 
			for index in range(0,len(en)-1):
				if en[index] == word:
					w_list.append(word)
			#print 'removed ' + word
	w_list = list(set(w_list))
	#deletion
	for w in w_list:
		if en.count(w) > 0:
			en.remove(w)
	w_list = []
	for word in word_features_zh:
		if (word in doc_words_zh): 
			for index in range(0,len(zh)-1):
				if zh[index] == word:
					w_list.append(word)
			#print 'removed ' + word
	#w_list = list(set(w_list))
	#deletion
	for w in w_list:
		zh.remove(w)
		#print zh
	'''
			
	######################
	'''#contain script
	for word_zh in word_features_zh:
		features['contains(%s)' % word_zh] = (word_zh in doc_words_zh)
	'''
	##################################
	##LexicalModelDict##
	# print '###Featuresets from LexicalModelDict###'
	for word_zh in zh:
		tmp_int = 0
		for word_en in en:
			tmp_int += count_match_dict(word_zh, word_en)
		features['num_match(%s)' % unicode(word_zh, 'utf-8')] = tmp_int
		'''
		if tmp_int == 0:
			features['num_match(%s)' % unicode(word_zh, 'utf-8')] = False
		else:
			features['num_match(%s)' % unicode(word_zh, 'utf-8')] = True
		'''
		# print 'num_match(%s) : %d' % (word_zh,tmp_int)
	# print '###Finished Featuresets from LexicalModelDict###'
	##
	##Matches Words_number##
	tmp_int = 0
	for word_zh in zh:
		for word_en in en:
			n_zh = unicodedata.normalize('NFKC', word_zh.decode('utf8'))
			n_en = unicodedata.normalize('NFKC', word_en.decode('utf8'))
			if n_zh == n_en:
				tmp_int += 1
				#features['match_word(%s)' % n_zh] = True
	#print tmp_int
	features['match_word'] = tmp_int
	#####
	# Unknown Words###can't find in lexical table
	'''
	tmp_int = 0
	for word_zh in zh:
		if count_search_dict(word_zh) == 0:
			tmp_int += 1
	features['unknown_word'] = tmp_int
	####'''
	return features
'''

'''


import uniout
import sys
import unicodedata


def append_features(corpus, tag, q, long, pid):
	print '\r###Appending features Tag : %5s' % tag + '#####PID : %2d' % (pid) + '###########           \r'

	run = 0
	len_all = len(corpus)
	featuresets = []
	for (zh, en) in corpus:
		if run % 100 == 0:
			tmp_per = float(float(run) / float(len_all)) * 100
			percent = "{0:.2f}".format(tmp_per)
			# sys.stdout.write('\n\r'*pid)
			# sys.stdout.flush()
			sys.stdout.write('\r'+' ' * int(long*1.7) + '-PID:%2d' % (pid) +
							 '-Line(%5s):' % tag + str(run) + '-' + str(percent) + '%           \r')
			sys.stdout.flush()
			# print '-Line('+tag+'):' + str(run) + '-',
		run = run + 1
		featuresets.append([gender_features(zh, en), tag])
	q.put(featuresets)
	print '\r###End of features Tag : %5s' % tag + '#####PID : %2d' % (pid) + '###########              \r'


def split_list(alist, wanted_parts=1):
	length = len(alist)
	return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
			for i in range(wanted_parts)]


def muti_feat_adder(cores=2, c_l=[]):
	featuresets = []
	import multiprocessing
	from multiprocessing import Process, Queue

	if cores % 2 != 0:
		print 'number of Cores must be even'
		return

	#if __name__ == '__main__':
	print '###Mutiprocessing cores = ' + str(cores) + '###'
	long = 0
	##
	q_list = []
	corpus_list = []
	jobs = []

	if len(c_l) != 1:
		numOfcuts = cores / 2
	else:
		numOfcuts = cores
	# if (c_l[1])
	for corpus in c_l:
		for each in split_list(corpus[0], numOfcuts):
			corpus_list.append([each, corpus[1]])
	# print corpus_list

	# single corpus

	for x in range(1, cores + 1):
		#print '###P : ' + str(x) + '###'
		tmp_q = Queue()
		q_list.append(tmp_q)
		####
		tmp_p = multiprocessing.Process(target=append_features, args=(
			corpus_list[x - 1][0], corpus_list[x - 1][1], tmp_q, long, x))
		long += 20
		if long > 20:
			# print '#######'
			long = 0
		jobs.append(tmp_p)
		tmp_p.daemon = False
		tmp_p.start()
	####
	index = 1
	for q in q_list:
		sys.stdout.write('\r-F_ID:%d-' % index + '\r')
		sys.stdout.flush()
		featuresets += q.get()
		del q
		index += 1
	for p in jobs:
		p.join()
		del p
		#q = Queue()
		#q2 = Queue()
		#p = multiprocessing.Process(target=append_features, args=(corpus_ok,"OK",q,long))
		# jobs.append(p)
		#p.daemon = False
		# p.start()
		# start 2
		#long += 20
		#p2 = multiprocessing.Process(target=append_features, args=(corpus_wrong,"WRONG",q2,long))
		# jobs.append(p2)
		#p2.daemon = False
		# p2.start()
		#featuresets += q.get()
		#del q
		# p.join()
		#featuresets += q2.get()
		#del q2
		# p2.join()
	print '##End of merging featureset##'

	return featuresets




def check_count(feat, tag):
	total = 0
	for node in feat:
		if node[1] == tag:
			total += 1
	print 'Total [' + tag + '] : ' + str(total)

#####################


# def print_maxent_test_header():
##


def test_maxent(algorithm,train_set,test_set):
	start_time = time.time()
	active_megam()
	print'%11s' % algorithm
	try:
		classifier
	except NameError:
		c_ex = True
	else:
		del classifier
	try:
		#global classifier
		classifier = nltk.classify.MaxentClassifier.train(
			train_set, algorithm, trace=1, max_iter=1000)
	except Exception as e:
		print 'Error: %r' % e
		return

	print 'This is most informative table'
	print classifier.show_most_informative_features(20)

	print 'Length of Testset :%d' % len(test_set)
	
	print 'Accuracy : ',
	print nltk.classify.accuracy(classifier, test_set)*100,
	print '%'

	print("---Total Used : %s Seconds ---" % (time.time() - start_time))
	
	return classifier
	# for featureset in test:
	#pdist = classifier.prob_classify(featureset)
	#print('%8.2f%6.2f' % (pdist.prob('x'), pdist.prob('y')) + end),
'''------------------------------------------------------------'''


def active_megam():
	if nltk.megam._megam_bin is None:
		import os
		path = os.getcwd()
		nltk.config_megam(path+'/megam/megam-64.opt')

#active_megam()
# print_maxent_test_header();
# test_maxent('GIS');
# test_maxent('IIS')
# nltk.config_megam('/home/db32555/MM/nltk/max/megam/megam-64.opt')
#classifier = test_maxent('MEGAM')
#classifier = test_maxent('TADM')
#import scipy
#classifier = test_maxent('LBFSGB')

# after we finished training


def corpus2():
	# corpus 2
	zh_dir_2 = 'zh-en/1m_uni.tok.zh'
	en_dir_2 = 'zh-en/1m_uni.tok.en'

	corpus_ok = read_corpus(en_dir_2, zh_dir_2, 'OK')
	corpus_wrong = read_corpus(en_dir_2, zh_dir_2, 'WRONG')
	corpus_wrong = shuffle_corpus(corpus_wrong)
	print '###Generating featuresets ctb###'
	n_c_l = [[corpus_ok, "OK"],[corpus_wrong, "WRONG"]]
	
	n_cores = int(args['cores'])
	
	f_ctb = muti_feat_adder(n_cores,n_c_l)
	random.shuffle(f_ctb)
	test_ctb = f_ctb  # all

	print 'Accuracy',
	print nltk.classify.accuracy(classifier, test_ctb)

#corpus2()
def sep_test_3(f_all,ratio_f_len):
	print 'Test_size : %.2f' % (ratio_f_len*100) + '%'
	#ratio_f_len = float(args['len_test_sets'])
	f_l_mid = int(len(f_all)*(ratio_f_len/3))
	f_l_end = len(f_all) - f_l_mid
	center = len(f_all) / 2
	test_ctb = f_all[:f_l_mid]
	test_ctb += f_all[center-f_l_mid/2:center+f_l_mid/2]
	test_ctb += f_all[f_l_end:]
	print 'Len of Test : %d ' % len(test_ctb)
	return test_ctb

def find_wrong(args,classifier):
	start_time = time.time()
	# corpus um
	print '####Function find_wrong####'
	zh_dir_2 = args['zh_dir']
	en_dir_2 = args['en_dir']
	#zh_dir_2 = 'zh-en/'+args['zh_dir']
	#en_dir_2 = 'zh-en/'+args['en_dir']
	#full_name = args['zh_dir'][:-3]
	full_name = args['output']
	full_name_tu = full_name + '.wrong'
	full_name_ta = full_name + '.ok'
	full_name_tm = full_name + '.mid'
	print '####OutputFile: wrong='+full_name_tu+' ok='+full_name_ta+'####'
	#zh_dir_2 = 'zh-en/test.zh'
	#en_dir_2 = 'zh-en/test.en'

	corpus_ok = read_corpus(en_dir_2, zh_dir_2, 'OK')
	n_c_l = [[corpus_ok, "OK"]]
	print '###Generating featuresets ctb###'
	
	n_cores = int(args['cores'])
	
	##lexical model
	global my_dict
	try:
		if my_dict is None:
			tmp_void = True
	except NameError:
		my_dict = []
		my_dict = prepare_lexical(my_dict,args['lex_table'])
	
	f_all = muti_feat_adder(n_cores, n_c_l)
	#corpus_wrong = read_corpus(en_dir_2,zh_dir_2,'WRONG')
	#corpus_wrong= shuffle_corpus(corpus_wrong)
	##
	#seaprate testset to 3 parts from total corpus
	test_ctb = sep_test_3(f_all,args['len_test_sets'])
	#
	print '###Staring calculate accuracy###'
	print 'Accuracy : ',
	print nltk.classify.accuracy(classifier, test_ctb)*100,
	print '%'
	
	print '###Finding Wrong & OK lines###'

	# remove tag from list
	# for index in range(0,len(f_all)):
	#	del f_all[index][1]
	# check each feature pro

	# preparefile
	#f = open('error/' + full_name + '_er', 'w')
	f_tu = open(full_name_tu, 'w')
	f_ta = open(full_name_ta, 'w')
	f_tm = open(full_name_tm, 'w')

	num_wrong = 0
	num_ok = 0
	num_mid = 0
	wrong_rate = float(args['wrong_rate'])
	ok_rate = float(args['ok_rate'])
	#
	run = 0
	for index in range(0, len(f_all)):
	
		if run % 100 == 0:
			tmp_per = float(float(run) / float(len(f_all))) * 100
			percent = "{0:.2f}".format(tmp_per)
			# sys.stdout.write('\n\r'*pid)
			# sys.stdout.flush()
			sys.stdout.write('\r'+'-Line(%5s):' %  str(run) + '-' + str(percent) + '%           \r')
			sys.stdout.flush()
			# print '-Line('+tag+'):' + str(run) + '-',
		run = run + 1
	
		##
	
		pdist = classifier.prob_classify(f_all[index][0])
		ok = float(pdist.prob('OK'))
		wrong = float(pdist.prob('WRONG'))
		if wrong >= wrong_rate:
			# print 'OK : ' + "{0:.3f}%".format(ok)
			# print 'WRONG : ' + "{0:.3f}%".format(wrong)
			# print 'Line : ' + str(index)
			f_tu.write('OK : ' + "{0:.3f}%".format(ok*100) + '\n')
			f_tu.write('WRONG : ' + "{0:.3f}%".format(wrong*100) + '\n')
			f_tu.write('Line : ' + str(index+1) + '\n')
			f_tu.write('Zh : '+ ' '.join(map(str, corpus_ok[index][1])))
			f_tu.write('En : '+ ' '.join(map(str, corpus_ok[index][0])))
			f_tu.write('--------------\n')
			num_wrong += 1
		if ok >= ok_rate:
			f_ta.write('OK : ' + "{0:.3f}%".format(ok*100) + '\n')
			f_ta.write('WRONG : ' + "{0:.3f}%".format(wrong*100) + '\n')
			f_ta.write('Line : ' + str(index+1) + '\n')
			f_ta.write('Zh : '+ ' '.join(map(str, corpus_ok[index][1])))
			f_ta.write('En : '+ ' '.join(map(str, corpus_ok[index][0])))
			f_ta.write('--------------\n')
			num_ok += 1
		if wrong < wrong_rate and ok < ok_rate:
			f_tm.write('OK : ' + "{0:.3f}%".format(ok*100) + '\n')
			f_tm.write('WRONG : ' + "{0:.3f}%".format(wrong*100) + '\n')
			f_tm.write('Line : ' + str(index+1) + '\n')
			f_tm.write('Zh : '+ ' '.join(map(str, corpus_ok[index][1])))
			f_tm.write('En : '+ ' '.join(map(str, corpus_ok[index][0])))
			f_tm.write('--------------\n')
			num_mid += 1
			
	print '\r',
	print 'Amount of wrong : ' + str(num_wrong)
	f_tu.write('Amount of wrong : ' + str(num_wrong))
	f_tu.close()

	print 'Amount of ok : ' + str(num_ok)
	f_ta.write('Amount of ok : ' + str(num_ok))
	f_ta.close()
	
	print 'Amount of mid : ' + str(num_mid)
	f_tm.write('Amount of mid : ' + str(num_mid))
	f_tm.close()
	
	print 'Total of lines : ' + str(len(f_all))
	
	print("---Total Used : %s Seconds ---" % (time.time() - start_time))
	
	return f_all

	
def sort_corpus(corpus):
	print '########Starting sort_corpus#######'
	##

	tmp = []
	for node in corpus:
		tmp.append(node[1])  # eng
	tmp.sort(key=lambda item: (-len(item), item),reverse=True)
	
	tmp_zh = []
	for node in corpus:
		tmp_zh.append(node[0])  # zh
	tmp_zh.sort(key=lambda item: (-len(item), item),reverse=True)
	##
	len_corpus = len(corpus)
	for index in range(0, len_corpus):
		corpus[index][1] = tmp[index]
		corpus[index][0] = tmp_zh[index]
	del tmp
	del tmp_zh
	print '########End of sort_corpus#######'
	return corpus
	
	
def print_corpus(corpus):
	full_name_zh = 'debug/hikari_zh'
	full_name_en = 'debug/hikari_en'
	f_zh = open(full_name_zh, 'w')
	f_en = open(full_name_en, 'w')
	for node in corpus:
		f_zh.write(' '.join(map(str, node[1])))
		f_en.write(' '.join(map(str, node[0])))
	f_en.close()
	f_zh.close()
	
def find_match(args,classifier):
	start_time = time.time()
	# find_match
	print '####Function find_match####'
	zh_dir_2 = args['zh_dir']
	en_dir_2 = args['en_dir']
	##
	full_name = args['output']
	print '####OutputFile='+full_name+'####'
	corpus_ok = read_corpus(en_dir_2, zh_dir_2, 'OK')
	#corpus_ok = shuffle_corpus(corpus_ok)
	sort = args['sort']
	if sort:
		corpus_ok = sort_corpus(corpus_ok);
	
	#for debug use only
	print_corpus(corpus_ok)
	#####
	print '###Generating Windowsize corpus###'
	
	corpus_size = len(corpus_ok)
	windows_size = int(float(args['win_size']*corpus_size))#5% of corpus as windows_size
	##
	print '###Windows Size : %d###' % windows_size
	
	real_corpus = []
	info_space = []
	
	tmp_info = 0
	for index in range(0, corpus_size):
		#record ori zh line
		zh_ori_line = []
		#got index then search
		#add original line
		tmp_info += 1
		real_corpus.append([corpus_ok[index][0],corpus_ok[index][1]])
		#
		zh_ori_line.append(index)
		for x in range(1, windows_size):
			front = index - x
			next = index + x
			if front > 0:
				tmp_info += 1
				real_corpus.append([corpus_ok[index][0],corpus_ok[front][1]])
				#
				zh_ori_line.append(front)
			if next < corpus_size:
				tmp_info += 1
				real_corpus.append([corpus_ok[index][0],corpus_ok[next][1]])
				#
				zh_ori_line.append(next)
		#for after searching use to detect f_sets position
		info_space.append([tmp_info,index,zh_ori_line])
		#import pdb
		#pdb.set_trace()
		#print info_space
		tmp_info = 0
		#debug
	
	
	n_c_l = [[real_corpus, "OK"]]
	
	#print real_corpus[0]
	#print real_corpus[1]
	#print real_corpus[2]
	#print real_corpus[11]
	#print real_corpus[12]
	
	print '###Generating featuresets ctb###'
	
	n_cores = int(args['cores'])
	
	##lexical model
	global my_dict
	try:
		if my_dict is None:
			tmp_void = True
	except NameError:
		#global my_dict
		my_dict = []
		my_dict = prepare_lexical(my_dict,args['lex_table'])
	
	f_all = muti_feat_adder(n_cores, n_c_l)
	##
	#'delete all tager'
	test_ctb = f_all  # all
	#
	print 'Accuracy(one to one) : ',
	print nltk.classify.accuracy(classifier, test_ctb)*100,
	print '%'
	############
	f = open(full_name, 'w')
	num_match = 0
	ok_rate = float(args['ok_rate'])
	corpus_size = len(f_all)
	#windows_size = int(float(args['win_size']*corpus_size))#5% of corpus as windows_size
	##
	print '####Starting search the highest OK lines####'
	run = 0
	
	try:
		#print wish_list
		wish_list.clear()
	except UnboundLocalError:
		wish_list = dict()

	cur_count = 0
	#
	#print info_space
	#print str(len(info_space))
	
	#print len(real_corpus)
	for index in range(0, corpus_size):
		#got index then search
		#run += 1
		now_count = info_space[0][0]-1
			#print now_count
		##
		#print 'c %d' % cur_count
		#print 'n %d' % now_count
		if cur_count == now_count:
			info_space.remove(info_space[0])
			#import pdb
			#pdb.set_trace()
			#print info_space
			#print str(len(info_space))
			cur_count = 0
			#
			change = True
			run += 1
		else:
			change = False
			cur_count += 1
		#else:
			#print 'round %d' % index

		#print 'done one line'
		if change:
			
			
			if wish_list:
				#empty wish_list
				#print 'empty'
				#continue
			
				#print wish_list
				# finish searching retrieve the highest matched line
				try:
					h_index = max(wish_list.iterkeys(), key=(lambda k: wish_list[k][0]))
					zh_line = info_space[0][2][wish_list[h_index][1]]
					#zh_line += 1
				except:
					break
				#print 'index : %d' % h_index
				#import pdb
				#pdb.set_trace()
				
				#print 'offset %d' % wish_list[h_index][1]
				#print h_index
				#print wish_list[h_index]
				#print info_space[0]
				
				#find ori_line
				
				
				
				f.write('OK : ' + "{0:.3f}%".format(wish_list[h_index][0]) + '\n')
				f.write('WRONG : ' + "{0:.3f}%".format(1-wish_list[h_index][0]) + '\n')
				f.write('Line(zh) : ' + str(zh_line) + '\n')
				f.write('Line(en) : ' + str(info_space[0][1]) + '\n')
				f.write('zh : '+ ' '.join(map(str, real_corpus[h_index][1])))
				f.write('en : '+ ' '.join(map(str, real_corpus[h_index][0])))
				f.write('--------------\n')
				num_match += 1
				wish_list.clear()
				change = False
		else:
			pdist = classifier.prob_classify(f_all[index][0])
			ok = float(pdist.prob('OK'))
			wrong = float(pdist.prob('WRONG'))
			
			#if index < 25 and index > 10:
			'''
			import pdb
			pdb.set_trace()
			print '---------'
			print f_all[index][0]
			print ' '.join(map(str, real_corpus[index][1]))
			print ' '.join(map(str, real_corpus[index][0]))
			print 'OK : %f' % ok
			'''
			#print 'ok : %f' % ok
			#print 'index : %d ' % index
			if ok >= ok_rate:
				#print f_all[index][0]
				#print ok
				offset = cur_count - 1
				#offset = cur_count + 1
				wish_list.update({index: [ok,offset]})

	print 'Amount of match : ' + str(num_match)
	f.write('Amount of match : ' + str(num_match))
	
	f.close()
	
	print("---Total Used : %s Seconds ---" % (time.time() - start_time))
	#return f_all
	
	
#tmp = um_corpus()

##return func_list
#func_list = [test_maxent,um_corpus]
#parameters = [featuresets,args]
#return [func_list,parameters]

####end of main
	

def prepare_fset(args):
	start_time = time.time()
	#####Reading Corpus From file#########
	#token_dir = 'data/max_token'
	#wrong_dir = 'data/max_token_wrong'
	###
	zh_dir = args['zh_dir']
	en_dir = args['en_dir']
	
	corpus_ok = read_corpus(en_dir, zh_dir, "OK")
	
	corpus_wrong = read_corpus(en_dir, zh_dir, 'WRONG')
	
	
	corpus_wrong = shuffle_corpus(corpus_wrong)
	
	print '########Finished Reading Corpus from a file#######'
	print '#########################################'
	###############################
	#0 is zh , 1 is en
	####feature############
	print '###Generating featuresets###'
	
	# seaprate eng & chinese
	######################
	# config
	freq_weight = 0.1
	##
	print '###Seprerating corpus(zh,eng) ###'
	global my_dict
	my_dict = []
	my_dict = prepare_lexical(my_dict,args['lex_table'])
	print '###Finished preaparing LexicalModelDict###'
	################################


	####frequest words####
	'''
	#####
	print '###Generating frequest words ###'

	words_en = []
	words_zh = []
	for node in corpus_ok:
		for a in node[0]:
			words_en.append(a)
		for b in node[1]:
			words_zh.append(b)

	all_words_en = nltk.FreqDist(w.lower() for w in words_en)
	all_words_zh = nltk.FreqDist(c for c in words_zh)

	word_features_en = list(sorted(all_words_en, key=all_words_en.__getitem__, reverse=True))
	word_features_zh = list(sorted(all_words_zh, key=all_words_zh.__getitem__, reverse=True))

	word_features_en = word_features_en[:50]
	word_features_zh = word_features_zh[:50]
	#word_features_en = sorted(all_words_en)[:int(len(all_words_en)*freq_weight)]
	#word_features_zh = sorted(all_words_zh)[:int(len(all_words_zh)*freq_weight)]
	'''
	######################
	'''prepare_lexical'''
	
	featuresets = []
	
	c_l = [[corpus_ok, "OK"], [corpus_wrong, "WRONG"]]
	#c_l = [[corpus_ok,"OK"]]

	n_cores = int(args['cores'])
	featuresets = muti_feat_adder(n_cores, c_l)

	#featureset += q.get()
	# p2.join()
	# print '2  '+q.get()


	###
	# append_features(corpus_ok,"OK")
	# append_features(corpus_wrong,"WRONG")

	import random
	random.shuffle(featuresets)
	#featuresets = [(gender_features(zh,en),"OK") for (zh,en) in corpus]
	ratio_f_len = float(args['len_test_sets'])
	f_len = int(len(featuresets)*ratio_f_len)
	train = featuresets[f_len:]
	#devtest = featuresets[1000:2000]
	test = featuresets[:f_len]

	print '\n###End of Generating featuresets###'
	######User Area#########
	
	print("---Total Used : %s Seconds ---" % (time.time() - start_time))
	
	parameters = args
	return [featuresets,train,test,parameters]
#if __name__ == "__main__":
	#main()

'''
ClassifierI supports the following operations:
  - self.classify(featureset)
  - self.classify_many(featuresets)
  - self.labels()
  - self.prob_classify(featureset)
  - self.prob_classify_many(featuresets)
  
 186 started : 12:07 ,End : 
'''
