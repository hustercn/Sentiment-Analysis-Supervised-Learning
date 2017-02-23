import sys
import collections
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        # print "Data loaded"
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        # print "Data processed"
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def add_word_count(list, isPositive):
    for sentence in list:
        for word in sentence:
            if word in count_dictionary:
                if isPositive:
                    count_dictionary[word][0] = count_dictionary[word][0] + 1
                else:
                    count_dictionary[word][1] = count_dictionary[word][1] + 1
            else:
                if isPositive:
                    count_dictionary[word] = [1, 0]
                else:
                    count_dictionary[word] = [0, 1]

def generate_features(positive_length, negative_length):
    index = 0
    for word, array in count_dictionary.items():
        positive_count = array[0]
        negative_count = array[1]
        if positive_count >= 0.01*positive_length or negative_count >= 0.01*negative_length:
            if positive_count == 0 or negative_count == 0 or positive_count >= 2*negative_count or negative_count >= 2*positive_count:
                features_map[word] = index
                index += 1

def construct_binary_vector(list):
    binary_vector = [[0 for x in range(len(features_map.keys()))] for y in range(len(list))]
    for i in range(len(list)):
        for word in list[i]:
            if word in features_map.keys():
                binary_vector[i][features_map[word]] = 1
    return binary_vector

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk

    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    #train_pos
    for i in range(len(train_pos)):
        train_pos[i] = [word for word in train_pos[i] if word not in stopwords]

    #train_neg
    for i in range(len(train_neg)):
        train_neg[i] = [word for word in train_neg[i] if word not in stopwords]

    #test_pos
    for i in range(len(test_pos)):
        test_pos[i] = [word for word in test_pos[i] if word not in stopwords]

    #test_neg
    for i in range(len(test_neg)):
        test_neg[i] = [word for word in test_neg[i] if word not in stopwords]
    
    global count_dictionary
    count_dictionary = {}

    add_word_count(train_pos, True)
    add_word_count(train_neg, False)
    # print "Dict len " + str(len(count_dictionary))

    global features_map
    features_map = {}

    generate_features(len(train_pos), len(train_neg))
    # print "Feature len " + str(len(features_map.keys()))

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = construct_binary_vector(train_pos)
    # print "got list 1"
    train_neg_vec = construct_binary_vector(train_neg)
    # print "got list 2"
    test_pos_vec = construct_binary_vector(test_pos)
    # print "got list 3"
    test_neg_vec = construct_binary_vector(test_neg)
    # print "got list 4"

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    global tag_train_pos
    tag_train_pos = "TAG_TRAIN_POS_"
    global tag_train_neg
    tag_train_neg = "TAG_TRAIN_NEG_"
    global tag_test_pos
    tag_test_pos = "TAG_TEST_POS_"
    global tag_test_neg
    tag_test_neg = "TAG_TEST_NEG_"

    labeled_train_pos = []
    for i in range(len(train_pos)):
        labeled_train_pos.append(LabeledSentence(words=train_pos[i], tags=[tag_train_pos+str(i)]))

    labeled_train_neg = []
    for i in range(len(train_neg)):
        labeled_train_neg.append(LabeledSentence(words=train_neg[i], tags=[tag_train_neg+str(i)]))

    labeled_test_pos = []
    for i in range(len(test_pos)):
        labeled_test_pos.append(LabeledSentence(words=test_pos[i], tags=[tag_test_pos+str(i)]))

    labeled_test_neg = []
    for i in range(len(test_neg)):
        labeled_test_neg.append(LabeledSentence(words=test_neg[i], tags=[tag_test_neg+str(i)]))

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = [], [], [], []
    for val in model.docvecs.doctags.keys():
        vec = train_pos_vec
        if tag_train_neg in val:
            vec = train_neg_vec
        elif tag_test_pos in val:
            vec = test_pos_vec
        elif tag_test_neg in val:
            vec = test_neg_vec
        vec.append(model.docvecs[val])

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    gnb = BernoulliNB(alpha=1.0, binarize=None)
    nb_model = gnb.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    lr = LogisticRegression()
    lr_model = lr.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    gnb = GaussianNB()
    nb_model = gnb.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    lr = LogisticRegression()
    lr_model = lr.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    tp, fp, tn, fn = 0, 0, 0, 0
    Y = np.array(["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec))
    predicted_values = model.predict(np.array(test_pos_vec + test_neg_vec))
    
    for i in range(len(Y)):
        if predicted_values[i] == Y[i] == 'pos':
            tp += 1
        if predicted_values[i] == Y[i] == 'neg':
            tn += 1
        if predicted_values[i] == 'pos' and Y[i] == 'neg':
            fp += 1
        if predicted_values[i] == 'neg' and Y[i] == 'pos':
            fn += 1

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    accuracy = 1.0*(Y == predicted_values).sum()/len(Y)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
