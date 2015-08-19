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