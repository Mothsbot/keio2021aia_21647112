import pickle
import numpy
import sys

infile = open("cat_data.pkl", 'rb')
new_dict = pickle.load(infile)
infile.close

numpy.set_printoptions(threshold=sys.maxsize)
print(new_dict)
