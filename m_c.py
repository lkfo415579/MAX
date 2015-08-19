# encoding=utf-8
#max_controller
import max

import pre_recall
######change process name##
import sys
if sys.platform == 'linux2':
    import ctypes
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    libc.prctl(15, 'Max_:>', 0, 0, 0)

######default area#########
default_max =  {'zh_dir':'zh-en/test.zh','en_dir':'zh-en/test.en','cores':'12','lex_table':'zh-en/zh2e_f','len_test_sets':0.02}
default_lex =  {'filename':'zh-en/zh2e','less_than':0.01,'order':'e2f','top':False,'symbol':True}
default_find = {'zh_dir':'zh-en/1000.test.zh','en_dir':'zh-en/1000.test.en','wrong_rate':0.95,'ok_rate':0.95,'cores':12,'output':'error/1000.test','lex_table':'zh-en/zh2e_f','len_test_sets':0.07}
default_match = {'zh_dir':'zh-en/1000.test.zh','en_dir':'zh-en/1000.test.en','ok_rate':0.95,'cores':12,'output':'match/1000.test.match','lex_table':'zh-en/zh2e_f','win_size':0.05,'sort':True}
default_pre =  {'f_sets_file':'object/f_sets.p','class_file':'object/class.p'}
default_save = {'class_output_file':'object/class.p','f_sets_output_file':'object/f_sets.p','id_f':0}
default_list = {'prepare':default_max,'lex_fiter':default_lex,
'find_wrong':default_find,'pre':default_pre,'save':default_save}
###########################
none = None
vars = []
paras = []
#####
def h():
	import subprocess
	subprocess.call("cat HELP", shell=True)
#display help usage
if __name__ == "__main__":
	h()
def defaultor(arg_input,default_set):
	args = {}
	args = default_set;
	for key in default_set:
		try: 
			args[key] = arg_input[key]
		except KeyError:
			args[key] = default_set[key]
	print args
	return args

def prepare(arg_input={}):
	
	args = defaultor(arg_input,default_max)
	
	data = max.prepare_fset(args)
	global vars
	vars.append(data[0])
	global paras
	paras.append(data[3])
	print 'ID of featureset is : %d' % (len(vars) - 1)
	#train set , test set
	###################
	global train_set
	train_set = data[1]
	global test_set
	test_set = data[2]
	
	
def p(arg_input={}):
	prepare(arg_input)

def train():
	global classifier
	classifier = max.test_maxent('MEGAM',train_set,test_set)

def t():
	train()

def e():
	exit()
def one(arg_input={}):
	p(arg_input)
	t()
def auto(arg_input={}):
	one(arg_input)


def lex_fiter(arg_input={}):
	import lex_f
	args = defaultor(arg_input,default_lex)
	lex_f.lex_f_main(args)

def l(arg_input={}):
	lex_fiter(arg_input)



def find_match(arg_input={}):
	args = defaultor(arg_input,default_match)
	max.find_match(args,classifier)

def find_wrong(arg_input={}):
	args = defaultor(arg_input,default_find)
	max.find_wrong(args,classifier)

def load(arg_input={}):
	import pickle
	args = defaultor(arg_input,default_pre)
	global classifier
	load_data = pre_recall.pre_recall(args)
	if load_data[0] != 'no':
		classifier = load_data[0]
	global vars
	if load_data[1] != 'no':
		vars.append(load_data[1])

def save(arg_input={}):
	import pickle
	args = defaultor(arg_input,default_save)
	if args['class_output_file'] != 'no':
		pickle.dump( classifier, open( args['class_output_file'], "wb" ))
	featuresets = vars[args['id_f']]
	if args['f_sets_output_file'] != 'no':
		pickle.dump( featuresets, open( args['f_sets_output_file'], "wb" ))

def TFPN(len=1000,id_f=0):
	dev = vars[id_f][:len]
	pre_recall.TFPN_TABLE(classifier,dev)
	
def default():
	for (key,val) in default_list.iteritems():
		print "Name(%s): " % key,
		print val
'''Older Code'''
''' *args
for x in range(0,len(default_lex)):
	if arg_input[x] is None:
		args[x] = default_lex[x]
	else:
		args[x] = arg_input[x]'''

