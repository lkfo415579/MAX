import cPickle as pickle
#pickle.dump( favorite_color, open( "class.p", "wb" ) )
classifier = pickle.load( open( "class.p", "rb" ) )
f_sets = pickle.load(open('f_sets.p','rb'))
#################
import nltk