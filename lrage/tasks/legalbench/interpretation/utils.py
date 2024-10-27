import sklearn.metrics
import numpy as np
from nltk.stem.porter import *
import string


def balanced_acc(items):
    print("balancedacc called")
    return items

def balanced_acc_agg(items):
    print("agg called")
    answers = list(zip(*items))[0]
    generations = list(zip(*items))[1]
    generations = [np.argmax(g) for g in generations]
    return sklearn.metrics.balanced_accuracy_score(answers, generations)