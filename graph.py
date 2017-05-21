import re
from os import listdir
from stop_words import get_stop_words
import numpy as np

stop_words = get_stop_words('english')

# lower and tokenize
def tokenize(corpus):
    all_words = re.findall(r'\w+', corpus.lower())
    return all_words

# add dictionary if not exists.
def add_dictionary(word, dictionary):
    if not(word is dictionary):
        dictionary[word] = 0


def create_dictionary(path, dictionary):
    for file in listdir(path):
        if file.endswith(".txt"):
            # only txt files
            corpus = open(path+'/'+file).read()
            word_list = tokenize(corpus)
            for word in word_list:
                    add_dictionary(word, dictionary)


# hash all words for accesing easily.
def index_dict(dictionary):
    counter = 0;
    inv_dict = []
    for key in dictionary:
        dictionary[key] = counter
        inv_dict.append(key)
        counter += 1
    return inv_dict


def create_graph(path, dictionary):
    create_dictionary(path,dictionary)
    ## Initial parameters.
    window_size = 10;
    interval = int(window_size/2);
    size = len(dictionary)
    ###
    address_map = index_dict(dictionary)
    t_matrix = np.zeros((size, size), dtype=np.float32)
    for file in listdir(path):
        if file.endswith(".txt"):
            corpus = open(path+'/'+file).read()
            word_list = tokenize(corpus)
            word_len = len(word_list)
            for w in range(0, word_len):
                word = word_list[w]
                low_range = w - interval
                if low_range < 0:
                    low_range = 0
                high_range = w + interval
                if high_range > word_len-1:
                    high_range = word_len-1
                for i in range(low_range, high_range+1):
                    target = dictionary[word]
                    match_word = word_list[i]
                    match = dictionary[match_word]
                    t_matrix[target][match] += 1
    return [t_matrix, address_map]


def learn(path, dictionary):
    # Initial parameters
    random_jump = 0.1
    top = 100
    eps = 0.0001
    [matrix, address_map] = create_graph(path, dictionary)
    size = np.shape(matrix)[0]
    # normalize the matrix
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    # add random walk probability.
    matrix *= (1-random_jump)
    matrix += random_jump / size
    # initial vector is defined.
    initial_vector = np.random.randn(size)
    initial_vector = np.abs(initial_vector)
    # normalized the vector.
    initial_vector = initial_vector / np.linalg.norm(initial_vector)
    print("Iteration Starts")
    for i in range(200):
        # iteration
        initial_vector_x = initial_vector.dot(matrix)
        print("Iteration :", i)
        # if norm-2 difference is below eps halt.
        if np.linalg.norm(initial_vector_x-initial_vector) < eps:
            print("Converged.")
            break
        initial_vector = initial_vector_x
    # sort top to bottom order.
    index = initial_vector.argsort()[::-1][:size]
    c = 0;
    print("---Words----")
    for i in range(size):
        word = address_map[index[i]];
        ## elimitate stop words.
        if word in stop_words:
            continue
        print(word)
        c = c + 1
        if c == top:
            print("------------")
            break


