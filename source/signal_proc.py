import pandas as pd
import numpy as np


def car_filter(df):
    """
    Applies the Common Average Reference filter to a dataset of subjects.
    @param df pandas series with index equals to the subjects
              every entry is a pandas series with index equals to the states
              of stimuli where every entry is a 2d (channel, time) numpy array;
    @return same structure with filter applied.
    """

    # initialize result
    new = pd.Series(index=df.index, dtype='object')

    for subject in df.index:

        # initialize result
        new[subject] = pd.Series(index=df[subject].index, dtype='object')

        for freq in df[subject].index:

            # initialize result
            new[subject][freq] = \
                np.zeros(df[subject][freq].shape, dtype='float64')

            # the new value for each sample is itself minus the average of all channels
            for i in range(df[subject][freq].shape[0]):
                new[subject][freq][i] = \
                    df[subject][freq][i] - df[subject][freq][i].mean()

    return new


def get_windows(df, window_time, f_samp):
    """
    Splits the data into windows of same size.
    @param df pandas series with index equals to the subjects
              every entry is a pandas series with index equals to the states
              of stimuli where every entry is a 2d (channel, time) numpy array;
    @return pandas dataframe with 3d (window, channel, time) numpy arrays.
    """

    # get windows length
    window_len = f_samp * window_time

    # initialize result
    new = pd.Series(index=df.index, dtype='object')

    for subject in df.index:

        # initialize result
        new[subject] = pd.Series(index=df[subject].index, dtype='object')

        for freq in df[subject].index:

            # get number of windows
            n_windows = int(df[subject][freq].shape[1] / window_len)

            # initialize result
            new[subject][freq] = np.zeros(
                (n_windows, df[subject][freq].shape[0], window_len),
                dtype='float64'
            )

            # split and store
            for i in range(n_windows):
                new[subject][freq][i] = \
                    df[subject][freq][:, i * window_len: (i + 1) * window_len]

    return new


def extract_attributes(df, states, f_samp, n_harmonics):
    """
    Extracts the amplitude of the Fourier Transform of the each signal in data.
    @param df pandas series with index equals to the states of stimuli
              every entry is a 3d (window, channel, time) numpy array;
    @return pandas dataframe with extracted attributes.
    """

    # get all freqs where attributes will be extracted from
    freqs = sorted(
        list(
            set(
                [j * (i + 1) for j in states for i in range(n_harmonics)]
            )
        )
    )

    # number of windows and channels from the first frequency should not be different for the others
    n_windows, n_channels, window_len = df[df.index[0]].shape

    # initialize result
    mat = pd.DataFrame(
        columns=[
            f'ch{channel}_{harmonic}Hz'
            for channel in range(n_channels)
            for harmonic in freqs
        ]
    )
    classes = []
    i = 0

    # extract attributes
    for state in states:

        # get fft for this whole state
        fft = np.abs(np.fft.fft(df[state]))

        # get corresponding frequencies
        freq_array = np.linspace(0, f_samp, window_len)

        for window in range(n_windows):

            # track which state is added in what order
            classes.append(state)

            # initialize result, row will one input
            row = {}

            for channel in range(n_channels):
                for harmonic in freqs:

                    # get the amplitude of the fft of the desired frequency 
                    row[f'ch{channel}_{harmonic}Hz'] = fft[
                        window,
                        channel,
                        np.where(
                            freq_array == min(
                                freq_array,
                                key=lambda x:abs(x - harmonic))
                        )[0][0]
                    ]
            
            # include result and go to next row
            mat.loc[i] = row
            i += 1

    # include output
    mat['state'] = classes

    # each column of output will actually be if that input corresponds or not to one state
    return pd.get_dummies(mat, columns=['state'])
