import numpy as np


def linear_discriminant_analysis(df):
    """
    Determine weights for Fischer linear discriminant analysis of df.
    @param df pandas dataframe with output in column 'state'
              states must be 1 or -1;
    @return list of weights and weight threshold.
    """

    # separate df in states
    group_by = df.groupby('state')

    # get means of classes
    mu1 = group_by.mean().loc[1].to_numpy(dtype='float64')
    mu2 = group_by.mean().loc[-1].to_numpy(dtype='float64')

    # get inputs
    x1 = group_by.get_group(1).drop('state', axis=1).to_numpy(dtype='float64')
    x2 = group_by.get_group(-1).drop('state', axis=1).to_numpy(dtype='float64')

    # get covariance matrices
    s1 = np.zeros((x1.shape[1], x1.shape[1]))
    s2 = np.zeros((x2.shape[1], x2.shape[1]))
    for i in range(x1.shape[0]):
        s1 += np.vstack(x1[i] - mu1) @ np.atleast_2d(x1[i] - mu1)
    for i in range(x2.shape[0]):
        s2 += np.vstack(x2[i] - mu2) @ np.atleast_2d(x2[i] - mu2)

    # get intra-class covariance matrix
    sw = s1 + s2

    # get weights
    w = np.hstack(np.linalg.pinv(sw) @ np.vstack(mu1 - mu2))

    # get threshold
    w0 = (w @ np.vstack(mu1) + w @ np.vstack(mu2)) / 2

    return w, w0[0]


def classify(df, w):
    """
    Classify result of linear discriminant analysis for different classifiers.
    @param df pandas dataframe;
    @param w dict[classifier: (list of weights, weight threshold)];
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
        y = x@wi[0]

        # append output
        new[f'lda_{classifier}'] = y - wi[1]

        # get states
        states = classifier.split('_')

        # append output
        new[f'lda_{classifier}_class'] = [
            states[0] if i > 0 else states[1] for i in y
        ]

    return new


def choose_class(df):
    """
    Choose class for each input in df based on classifications for
    different classifiers.
    @param df pandas dataframe with lda classified;
    @return df with appended result.
    """

    # get output columns
    lda_columns = [i for i in df.columns if 'lda_' in i and '_class' not in i]
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
            line = df[lda_columns].loc[i]
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
    chosen_df['lda_chosen'] = result

    return chosen_df
