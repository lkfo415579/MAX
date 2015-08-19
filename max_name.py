import nltk
nltk.usage(nltk.classify.ClassifierI)
###
import random
from nltk.corpus import names
names = ([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')] )
random.shuffle(names)

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
classifier = test_maxent('MEGAM')
#test_maxent('TADM')


