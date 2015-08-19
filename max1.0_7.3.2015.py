#encoding=utf-8
import nltk
nltk.usage(nltk.classify.ClassifierI)
###

#####Reading Corpus From file#########
token_dir = 'data/max_token'
wrong_dir = 'data/max_token_wrong'
print '########Reading Corpus From file "'+token_dir+'"###########'

f = open(token_dir,'r')

str_corpus = f.readline()

import ast

corpus_ok = ast.literal_eval(str_corpus)
####
print '########Reading Corpus From file "'+wrong_dir+'"###########'
##wrong corpus
f = open(wrong_dir,'r')

str_corpus = f.readline()

corpus_wrong = ast.literal_eval(str_corpus)

del str_corpus

print '########Finished Reading Corpus from a file#######'

###############################
#0 is zh , 1 is en
####feature############
print '###Generating featuresets###'

##seaprate eng & chinese
######################
##config
freq_weight = 0.1
##
print '###Seprerating corpus(zh,eng) ###'
#words_eng = []
#words_zh = []
'''
for node in corpus_ok:
	for a in node[1]:
		words_eng.append(a)
	for b in node[0]:
		words_zh.append(b)
'''
#all_words_eng = nltk.FreqDist(w.lower() for w in words_eng)
#all_words_zh = nltk.FreqDist(c for c in words_zh)

#word_features_eng = all_words_eng.keys()[:int(len(all_words_eng)*freq_weight)]
#word_features_zh = all_words_zh.keys()[:int(len(all_words_zh)*freq_weight)]

def gender_features(zh,en):
	features = {}
	#num of words
	features["num_eng"] = len(en)
	features["num_zh"] = len(zh)
	##freq_word_features##
	#doc_words_eng = set(en)
	#doc_words_zh = set(zh)	
	#run = 0
	#for word in word_features_eng:
	#	features['contains(%s)' % word] = (word in doc_words_eng)
	#for word_zh in word_features_zh:
	#	features['contains(%s)' % word_zh] = (word_zh in doc_words_zh)		
	###
	return features
'''

'''
featuresets = []
def append_features(corpus,tag):
	run = 0
	for (zh,en) in corpus:
		featuresets.append([gender_features(zh,en),tag])
		if run % 1000 == 0:
			print '---Line('+tag+') : ' + str(run) + '---',
		run = run +1

append_features(corpus_ok,"OK")
append_features(corpus_wrong,"WRONG")
 
random.shuffle(featuresets)
#featuresets = [(gender_features(zh,en),"OK") for (zh,en) in corpus]
train = featuresets[2000:]
devtest = featuresets[1000:2000]
test = featuresets[:1000]

print '\n###End of Generating featuresets###'
######User Area#########

def check_count(feat,tag):
	total = 0
	for node in feat:
		if  node[1] == tag:
			total += 1
	print 'Total ['+tag+'] : ' + str(total)

#####################
end=' '

def print_maxent_test_header():
	##

def test_maxent(algorithm):
	print'%11s' % algorithm, end
	try:
		classifier = nltk.classify.MaxentClassifier.train(
						train, algorithm, trace=0, max_iter=1000)
	except Exception as e:
		print 'Error: %r' % e
		return
	
	print 'This is most informative table'
	print classifier.show_most_informative_features(10)

	print 'Accuracy',
	print nltk.classify.accuracy(classifier,test)
	
	return classifier
		
	#for featureset in test:
		#pdist = classifier.prob_classify(featureset)
		#print('%8.2f%6.2f' % (pdist.prob('x'), pdist.prob('y')) + end),
'''------------------------------------------------------------'''	

#print_maxent_test_header();
#test_maxent('GIS');
#test_maxent('IIS')
#nltk.config_megam('/home/db32555/MM/nltk/max/megam/megam-64.opt')
#classifier = test_maxent('MEGAM')
#test_maxent('TADM')


