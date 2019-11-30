import datetime as dt
import itertools
from random import sample

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pathlib
import pickle
from sklearn.metrics import confusion_matrix

from Data.suspicious_accounts import suspicious_account_usernames_with_posts
from Tools.util import load_user_posts

DATAPATH = pathlib.Path(__file__).parents[1].joinpath('Data')

def plot_confusion_matrix(cm, fig, ax, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.invert_yaxis()


def plot_train_test_cm(Y_train, T_train, Y_test, T_test, savepath=None):

    cm = confusion_matrix(y_true=T_train, y_pred=Y_train)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Train results')
    plot_confusion_matrix(cm, fig, ax1, ['Normal', 'Suspicious'])

    cm = confusion_matrix(y_true=T_test, y_pred=Y_test)
    plot_confusion_matrix(cm, fig, ax2, ['Normal', 'Suspicious'])
    ax2.set_title('Test results')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot_user_creation(label='suspicious'):
    """ Plots hist of creation dates for suspicious or normaL accounts """

    if label == 'suspicious':
        df = pd.read_csv(DATAPATH.joinpath('suspicious_accounts.csv'), index_col='author')
        # there is one account made in 2010 which messes up the plot so we leave it out
        df = df.sort_values('created').iloc[1:]
    else:
        df = pd.read_csv(DATAPATH.joinpath('normal_accounts.csv'), index_col='author')
        df = df.sort_values('created')

    # we need to convert dates to matplotlib friendly date format with mdates
    created = [mdates.datestr2num(ts) for ts in df['created'].values]

    fig, ax = plt.subplots(1, 1)

    # create bins for each month spanning June 2015 to December 2016
    bins = [mdates.date2num(dt.datetime(2015, x, 1)) for x in range(6, 13)]
    bins += [mdates.date2num(dt.datetime(2016, x, 1)) for x in range(1, 13)]

    ax.hist(created, bins=bins)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
    fig.autofmt_xdate()
    ax.set_ylabel('Number of accounts created')
    ax.set_title('{} account creation'.format('Suspicious' if label == 'suspicious' else 'Normal'))
    plt.show()


def plot_user_activity(username, label='suspicious'):
    """ Plots hist of post activity for a given user or list of users """

    if type(username) is str:
        title = f'Post activity for {username}'
        username = [username]
    else:
        title = f'Post activity for {len(username)} {label} users'

    created = []
    for u in username:
        try:
            subs, comms = load_user_posts(u, label=label)
        except ValueError:
            continue
        # user_metadata = pd.read_csv(DATAPATH.joinpath('suspicious_accounts.csv'), index_col='author').loc[u]
        # print(f'User {u} created on {user_metadata["created"]}')

        # we need to convert dates to matplotlib friendly date format with mdates
        posttimes = []
        posttimes += list(subs['created'].values) if subs is not None else []
        posttimes += list(comms['created'].values) if comms is not None else []
        for ts in posttimes:
            try:
                parsed_ts = [mdates.datestr2num(ts)]
            except TypeError:
                parsed_ts = []

            created += parsed_ts

    fig, ax = plt.subplots(1, 1)

    # create bins for each month spanning June 2015 to December 2016
    bins = [mdates.date2num(dt.datetime(2015, x, 1)) for x in range(6, 13)]
    bins += [mdates.date2num(dt.datetime(2016, x, 1)) for x in range(1, 13)]

    ax.hist(created, bins=bins)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
    fig.autofmt_xdate()
    ax.set_ylabel('Number of posts made')
    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    # plot_user_creation('asdr')
    # plot_user_activity(suspicious_account_usernames_with_posts)
    # normal_account_usernames = pd.read_csv(DATAPATH.joinpath('normal_accounts.csv'))['author'].values
    # plot_user_activity(normal_account_usernames, label='normal')
    res = pickle.load(open(DATAPATH.joinpath('LR_w2v_results.p'), 'rb'))

    plot_train_test_cm(*res, savepath=DATAPATH.joinpath('LR_w2v_results.png'))
