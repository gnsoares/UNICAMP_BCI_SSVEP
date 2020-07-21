import numpy as np


def pearson(df):
    """
    Take a classifier and evaluate the pearson correlation coefficient between
    each attribute and the output.
    @param df pandas dataframe;
    @return dict with the sorted correlation coefficient for each attribute. 
    """

    # get output columns
    state_columns = [i for i in df.columns if 'state' in i]
    y = df[state_columns].to_numpy()

    # initialize result
    result = {key: {} for key in state_columns}

    # evaluate attributes
    for i in df.drop(state_columns, axis='columns'):
        x = df[i].to_numpy()

        for j, state in enumerate(state_columns):

            # evaluate coefficient and append to result
            u = np.sum( ( x - np.mean(x) ) * ( y[:, j] - np.mean(y[:, j]) ) )
            d = np.sqrt(
                np.sum( ( x - np.mean(x) )**2 ) *
                np.sum( ( y[:, j] - np.mean(y[:, j]) )**2)
            )
            result[state][i] = np.abs(u/d)

    # return dict with: key = attribute, value = corr coef
    if len(result.keys()) == 1:
        return result[list(result.keys())[0]]
    else:
        return result
