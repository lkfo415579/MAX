import cPickle as pickle
c_name = "class.p"
f_name = "f_sets.p"
print '########Starting '+c_name+'#######'
pickle.dump( classifier, open( c_name, "wb" ) )
print '########Starting '+f_name+'#######'
pickle.dump( featuresets,open( f_name, "wb" ) )
print '########END#######'