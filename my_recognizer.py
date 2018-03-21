"""
####### AIND Artificial Intelligence Nanodegree - Udacity #######
#---------------------------- Roma -----------------------------#
#
# HMM - Term 1 End Project
#
#---------------------------------------------------------------#
#################################################################
"""
import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    proba = []
    guesses = []
    X_lengths = test_set.get_all_Xlengths()
    for X, lengths in X_lengths.values():
        log_l = {} # Save the likelihood of a word
        max_score = float("-inf") # Save max score as recognizer iterates the list
        best_guess = None # Save best guess as recognizer iterates the list
        for word, model in models.items():
            try:
                # Score word using model
                word_score = model.score(X, lengths)
                log_l[word] = word_score

                if word_score > max_score:
                    max_score = word_score
                    best_guess = word
            except:
                # Unable to process word
                log_l[word] = float("-inf")

        guesses.append(best_guess)
        proba.append(log_l)

    return proba, guesses
