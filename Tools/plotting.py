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
            subs, comms = load_user_posts(u)
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


def plot_user_activity_distribution(filepath, label='suspicious'):
    """
    Plots distribution of posting acitivty for a given group of users
    Args:
        filepath: name of file
        label: suspicious or normal
    """

    df = pd.read_csv(DATAPATH.joinpath(filepath))
    print(df.describe())
    num_subs = df['submissions'].values
    num_comms = df['comments'].values
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    bin_edges = range(0, max(num_subs) + 50, 50)
    axes[0].hist(num_subs, bins=bin_edges)
    axes[0].set_xticks(bin_edges)
    axes[0].set_ylabel('Number of submissions')
    axes[0].set_title(f'Submission activity distribution for {len(df)} {label} users')
    bin_edges = range(0, max(num_comms) + 50, 50)
    axes[1].hist(num_comms, bins=bin_edges)
    axes[1].set_xticks(bin_edges)
    axes[1].set_ylabel('Number of comments')
    axes[1].set_title(f'Comment activity distribution for {len(df)} {label} users')

    fig.tight_layout()
    plt.show()


def boxplot_outlier_elimination(filepath, col_names):
    df = pd.read_csv(DATAPATH.joinpath(filepath), index_col=0)
    IQRs = []
    for col_name in col_names:
        print(df.head())
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQRs.append(Q3 - Q1)

    for i, col_name in enumerate(col_names):
        IQR = IQRs[i]
        # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
        df = df.query(f'(@Q1 - 1.5 * @IQR) <= {col_name} <= (@Q3 + 1.5 * @IQR)')
        # df.join(filtered, rsuffix='_filtered').boxplot()
        print(df.head(10))

    df.to_csv(DATAPATH.joinpath(filepath[:-4] + '_filtered.csv'))


def plot_user_posting_times(users):
    """ Plots hist of posting times for a given user or list of users """

    if type(users) is str:
        title = f'Daily post activity for {users}'
        users = [users]
    else:
        title = f'Daily post activity for {len(users)} users'

    all_posttimes = []
    for u in users:
        try:
            subs, comms = load_user_posts(u)
        except ValueError:
            continue
        posttimes = []
        posttimes += list(subs['created'].values) if subs is not None else []
        posttimes += list(comms['created'].values) if comms is not None else []
        for ts in posttimes:
            try:
                parsed_ts = [int(ts.split()[1][:2])]
            except Exception as e:
                print(e)
                parsed_ts = []
            all_posttimes += parsed_ts

    fig, ax = plt.subplots(1, 1)

    # create bins for each hour of the day
    bins = range(25)

    ax.hist(all_posttimes, bins=bins)
    ax.set_xticks(list(bins)[:-1])
    ax.set_xlabel('Hour of day (UTC)')
    ax.set_ylabel('Number of posts made')
    ax.set_title(title)
    plt.show()


def sample_normal_accounts(save=False):
    ndf = pd.read_csv(DATAPATH.joinpath('large_unfiltered_NAs_final_filtered.csv'), index_col=0)
    sdf = pd.read_csv(DATAPATH.joinpath('suspicious_accounts.csv'), index_col=0)
    print(ndf.describe())
    print(sdf.describe())
    ndf_selected = pd.concat([ndf.loc[ndf['comments'] == 0, :],
                              ndf.loc[ndf['comments'] == 1, :],
                              ndf.loc[ndf['comments'] == 2, :]])
    ndf_selected = pd.concat([ndf_selected,
                              ndf.loc[ndf['comments'] > 1, :].sample(276 - len(ndf_selected))])
    fig, axes = plt.subplots(1, 4)
    sdf.boxplot(ax=axes[0])
    ndf.boxplot(ax=axes[1])
    sdf.boxplot(ax=axes[2])
    ndf_selected.boxplot(ax=axes[3])
    print(ndf_selected.describe())
    axes[0].set_ylim(0, 500)
    axes[1].set_ylim(0, 500)
    axes[2].set_ylim(0, 200)
    axes[3].set_ylim(0, 200)
    plt.show()
    if save:
        ndf_selected.to_csv(DATAPATH.joinpath('small_filtered_NAs_final.csv'))
    # print(ndf_selected.sort_values('submissions').tail(20))


if __name__ == '__main__':
    # plot_user_creation('asdr')
    # plot_user_activity(suspicious_account_usernames_with_posts)
    # normal_account_usernames = pd.read_csv(DATAPATH.joinpath('normal_accounts.csv'))['author'].values
    # plot_user_activity(normal_account_usernames, label='normal')

    # res = pickle.load(open(DATAPATH.joinpath('LR_w2v_results.p'), 'rb'))
    # plot_train_test_cm(*res, savepath=DATAPATH.joinpath('LR_w2v_results.png'))
    # plot_user_activity_distribution('normal_accounts2_filtered.csv', label='normal')
    # plot_user_activity_distribution('suspicious_accounts.csv', label='suspicious')

    # normal_account_usernames = pd.read_csv(DATAPATH.joinpath('normal_accounts.csv'))['author'].values
    # plot_user_posting_times(normal_account_usernames, 'normal')
    # plot_user_posting_times('Argeus', 'normal')
    # boxplot_outlier_elimination('large_unfiltered_NAs_final.csv', col_names=['submissions', 'comments'])
    sample_normal_accounts(save=True)
