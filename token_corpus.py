#encoding=utf-8
import nltk
nltk.usage(nltk.classify.ClassifierI)
###
import random
#from nltk.corpus import names
#names = ([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')] )
#random.shuffle(names)

'''---------'''
'''
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
corpusdir = 'zh-en/' # Directory of corpus.
newcorpus = PlaintextCorpusReader(corpusdir, '.*', encoding='utf-8')
'''
'''---------'''
'''
pa_zh = newcorpus.sents('test.zh')
a = 0

for object in pa_zh:
	for word in object:
		print word,
	print "OK"
	a = a + 1
'''
#####Read Corpus from paracorpus#########
def number_file(dir):
	with open(dir) as f:
		return sum(1 for _ in f)

en_dir = 'zh-en/test.en'
zh_dir = 'zh-en/test.zh'
en = open(en_dir,'r')
zh = open(zh_dir,'r')

len_p = number_file(en_dir)

corpus = []
for x in range(0, len_p):
	corpus.append([en.readline().strip(),zh.readline().strip()])

###############################

###tokenize chinese####
print '########Starting tokenization###########\n'
##
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
pre_path = '/home/db32555/MM/stanford-segmenter/'
segmenter = StanfordSegmenter(path_to_jar=pre_path+'stanford-segmenter-3.4.1.jar', path_to_sihan_corpora_dict=pre_path+'./data', path_to_model=pre_path+'./data/pku.gz', path_to_dict=pre_path+'./data/dict-chris6.ser.gz')
##setup_end##
from nltk import word_tokenize
##setup_end_eng##
for node in corpus:
	index = corpus.index(node)
	chinese = node[1]
	chinese = unicode(chinese, 'utf-8')
	tmp_segmented = segmenter.segment(chinese)
	tmp_segmented = tmp_segmented.split(" ")
	#
	del corpus[index][1]
	corpus[index].append(tmp_segmented)	
	print tmp_segmented
	##this is chinese 
	english = node[0]
	english = unicode(english, 'utf-8')
	english = word_tokenize(english)
	del corpus[index][0]
	corpus[index].append(english)
	print english
	##this is english
	
print '########End of tokenization###########\n'
print 'var is "corpus"'
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
'''
def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    features['suffix2'] = name[-2:].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features


featuresets = [(gender_features2(n),g) for (n,g) in names]
train = featuresets[1500:]
devtest = featuresets[500:1500]
test = featuresets[:500]
'''

'''
train = [
(dict(a=1,b=1,c=1), 'y'),
(dict(a=1,b=1,c=1), 'x'),
(dict(a=1,b=1,c=0), 'y'),
(dict(a=0,b=1,c=1), 'x'),
(dict(a=0,b=1,c=1), 'y'),
(dict(a=0,b=0,c=1), 'y'),
(dict(a=0,b=1,c=0), 'x'),
(dict(a=0,b=0,c=0), 'x'),
(dict(a=0,b=1,c=1), 'y'),
]
test = [
(dict(a=1,b=0,c=1)), # unseen
(dict(a=1,b=0,c=0)), # unseen
(dict(a=0,b=1,c=1)), # seen 3 times, labels=y,y,x
(dict(a=0,b=1,c=0)), # seen 1 time, label=x
]
'''

#####################
end=' '

def print_maxent_test_header():
	print(' '*11+''.join(['      test[%s]  ' % i
						  for i in range(len(test))]))
	print(' '*11+'     p(x)  p(y)'*len(test))
	print('-'*(11+15*len(test)))

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
#classifier = test_maxent('MEGAM')
#test_maxent('TADM')


