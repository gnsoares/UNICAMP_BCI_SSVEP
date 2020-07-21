import numpy as np
import pandas as pd


def linear_regression(df):
    """
    Determine weights for linear regression of df.
    @param df pandas dataframe with output in column 'state'
              states must be 1 or -1;
    @return list of weights.
    """

    # split input and output
    x = df.drop('state', axis=1).to_numpy(dtype='float64')
    y = list(df['state'].to_numpy(dtype='int8'))

    # get weights
    w = np.linalg.pinv(x)@y

    return w


def apply_regression(df, w):
    """
    Apply regression for different classifiers.
    @param df pandas dataframe;
    @param w dict[classifier: list of weights];
    @return df with appended result.
    """

    # get input
    if 'state' in df.columns:
        x = df.drop('state', axis=1).to_numpy(dtype='float64')
    else:
        x = df.to_numpy(dtype='float64')

    # initialize result
    new = df.copy()

    for classifier, wi in w.items():

        # evaluate output
        y = x@wi

        # append output
        new[f'lms_{classifier}'] = y

    return new


def classify(df):
    """
    Classify result of regression for different classifiers.
    @param df pandas dataframe with regression applied;
    @return df with appended result.
    """

    # get output columns
    lms_columns = [i for i in df.columns if 'lms_' in i]

    # get classifiers
    classifiers = [i[4:].split('_') for i in lms_columns]

    # classify
    result = {
        f'{c[0]}_{c[1]}': [
            c[0] if df.at[i, column] > 0 else c[1] for i in df.index
        ] for c, column in zip(classifiers, lms_columns)
    }

    # initialize result and append output
    new = df.copy()
    for k, v in result.items():
        new[f'lms_{k}_class'] = v

    return new


def choose_class(df):
    """
    Choose class for each input in df based on classifications for
    different classifiers.
    @param df pandas dataframe with regression classified;
    @return df with appended result.
    """

    # get output columns
    lms_columns = [i for i in df.columns if 'lms_' in i and '_class' not in i]
    class_columns = [i for i in df.columns if '_class' in i]

    # initialize result
    result = []

    for i in df.index:

        # get mode between classifiers
        mode = df[class_columns].loc[i].mode()

        # mode is unique: grab value
        if len(mode) == 1:
            chosen = mode[0]

        # mode is not unique: grab value of highest classifier
        else:
            line = df[lms_columns].loc[i]
            m = line.abs().max()
            for j, k in line.items():
                if abs(k) == m:
                    chosen_column = j
                    break
            states = chosen_column.split('_')
            chosen = states[1] if df.at[i, chosen_column] > 0 else states[2]

        result.append(chosen)

    # append result
    chosen_df = df.copy()
    chosen_df['lms_chosen'] = result

    return chosen_df
