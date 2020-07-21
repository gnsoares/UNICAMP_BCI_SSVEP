from source.lda import linear_discriminant_analysis
from source.lda import classify as LDAclassify
from source.lda import choose_class as LDAchoose_class
from source.lms import linear_regression
from source.lms import apply_regression
from source.lms import classify as LMSclassify
from source.lms import choose_class as LMSchoose_class
from source.ml import random_setify, generate_1x1_classifiers, drop_states


def __get_sets(df, states, states_to_drop, train_pct, valid_pct):
    """
    Auxiliary function that devides data into sets and format it in
    order to apply methods of predicition.
    @param df pandas dataframe;
    @param states states to be classified against each other;
    @param states_to_drop states not needed;
    @param train_pct percentage of samples in training set;
    @param valid_pct percentage of samples in validation set;
    @return (dict[states: classifier], validation df).
    """
    # split data into sets
    training, validation, testing = random_setify(df, train_pct, valid_pct)

    # split training set into 1x1 classifiers
    classifiers = generate_1x1_classifiers(training, states)

    # drop not needed states
    validation = drop_states(validation, states_to_drop)

    return classifiers, validation


def lms(df, states, states_to_drop, train_pct, valid_pct, repetitions):
    """
    Evaluate accuracy of least mean squares.
    @param df pandas dataframe;
    @param states states to be classified against each other;
    @param states_to_drop states not needed;
    @param train_pct percentage of samples in training set;
    @param valid_pct percentage of samples in validation set;
    @param repetitions times method will be applied;
    @return accuracy.
    """
    # initialize result
    accuracy = 0

    for _ in range(repetitions):

        # get sets
        classifiers, validation = __get_sets(df,
                                             states,
                                             states_to_drop,
                                             train_pct,
                                             valid_pct)

        # introducing bias
        for k, classifier in classifiers.items():
            state = classifier['state']
            classifier = classifier.drop('state', axis=1)
            classifier['w0'] = classifier.shape[0]*[1]
            classifier['state'] = state

        # apply method of classification
        w = {s: linear_regression(c) for s, c in classifiers.items()}
        classified = LMSclassify(apply_regression(validation, w))
        chosen = LMSchoose_class(classified)

        for i in chosen.index:

            # get expected and predicted states
            expected = int(chosen.at[i, 'state'])
            predicted = int(chosen.at[i, 'lms_chosen'])

            # update accuracy
            accuracy += 1/chosen.shape[0] if expected == predicted else 0
    
    return accuracy/repetitions


def lda(df, states, states_to_drop, train_pct, valid_pct, repetitions):
    """
    Evaluate accuracy of least mean squares.
    @param df pandas dataframe;
    @param states states to be classified against each other;
    @param states_to_drop states not needed;
    @param train_pct percentage of samples in training set;
    @param valid_pct percentage of samples in validation set;
    @param repetitions times method will be applied;
    @return accuracy.
    """
    # initialize result
    accuracy = 0

    for _ in range(repetitions):

        # get sets
        classifiers, validation = __get_sets(df,
                                             states,
                                             states_to_drop,
                                             train_pct,
                                             valid_pct)

        # apply method of classification
        w = {s: linear_discriminant_analysis(c) for s, c in classifiers.items()}
        classified = LDAclassify(validation, w)
        chosen = LDAchoose_class(classified)

        for i in chosen.index:

            # get expected and predicted states
            expected = int(chosen.at[i, 'state'])
            predicted = int(chosen.at[i, 'lda_chosen'])

            # update accuracy
            accuracy += 1/chosen.shape[0] if expected == predicted else 0
    
    return accuracy/repetitions
