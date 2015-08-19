def TFPN_TABLE(classifier,dev):
	errors = []
	all =[]
	for (f,tag) in dev:
		guess = classifier.classify(f)
		if  guess != tag:
			errors.append((tag,guess,f))
		all.append((tag,guess,f))

	positive = 'OK'
	negative = 'WRONG'
	
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	
	for (tag,guess,f) in all:
		if tag == guess and tag == positive:
			tp += 1
		if tag != guess and tag == positive:
			fp += 1
		if tag != guess and tag == negative:
			fn += 1
		if tag == guess and tag == negative:
			tn += 1
			
	print ' '*21+'Correct'+' '*5 + 'In-correct'
	print 'Selected'+' '*9 + 'True Positive' + ' ' *5 + 'False Positive'
	print ' '*19 + str(tp) + ' '*8 + str(fp)
	print 'Not Selected'+' '*5 + 'False Negative' + ' ' *5 + 'True Negative'
	print ' '*19 + str(fn) + ' '*8 + str(tn)
	
	return errors