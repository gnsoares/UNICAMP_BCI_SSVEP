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


def wrappers(df, evaluate_function, evaluate_args, forwards=True):
    """
    Take a classifier and evaluate the pearson correlation coefficient between
    each attribute and the output.
    @param df pandas dataframe;
    @param df evaluate_function function that returns accuracy;
    @param df evaluate_args all args of evaluate_function aside from df;
    @param df forwards True if forwards selection / False if backwards selection;
    @return dict with the selected attributes for each state. 
    """

    # get output columns
    state_columns = [i for i in df.columns if 'state' in i]
    y = df[state_columns].to_numpy()

    # get attributes
    all_attributes = [i for i in df.columns if i not in state_columns]

    # initialize result
    result = {key: [] for key in state_columns}

    # evaluate attributes
    for i, state in enumerate(state_columns):

        # initialize variables
        attributes_to_evaluate = all_attributes.copy()
        picked_attributes = []
        last_accuracy = -1

        # at most, wrappers will evaluate all attributes
        for _ in range(len(all_attributes)):

            # initialize accuracies
            accuracies = {}

            # evaluate all remaining attributes
            for j in attributes_to_evaluate:

                # select attributes to be dropped
                attributes_to_drop = attributes_to_evaluate.copy()
                attributes_to_drop.remove(j)

                # evaluate for current set of attributes
                accuracies[j] = evaluate_function(
                    df.drop(attributes_to_drop, axis=1),
                    *evaluate_args
                )

            # get attribute that returns maximum accuracy
            attribute, accuracy = max(accuracies.items(), key=lambda x: x[1])

            # accuracy improved: register and continue
            if accuracy > last_accuracy:
                picked_attributes.append(attribute)
                attributes_to_evaluate.remove()
                last_accuracy = accuracy

            # accuracy not improved: stop
            else:
                break

        # append result
        result[state] = picked_attributes

    # return dict with: key = attribute, value = list of picked attributes
    if len(result.keys()) == 1:
        return result[list(result.keys())[0]]
    else:
        return result