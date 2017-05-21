from graph import learn
import os
current_dir = os.getcwd()
path_to_neg = current_dir + "/data/neg"
path_to_pos = current_dir + "/data/pos"
neg_dictionary = {}
pos_dictionary = {}

print("--Negative--")
neg_t = learn(path_to_neg, neg_dictionary)
print("--Positive--")
pos_t = learn(path_to_pos, pos_dictionary)
