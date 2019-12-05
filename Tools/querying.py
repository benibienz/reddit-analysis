import datetime as dt
import os
import pandas as pd
import numpy as np
from time import sleep
from psaw import PushshiftAPI
from pprint import pprint
from dateutil import tz
import pathlib
import praw
from prawcore.exceptions import NotFound

from Data.suspicious_accounts import suspicious_account_usernames, suspicious_account_usernames_with_posts
from Tools.util import random_date, load_user_posts

DATAPATH = pathlib.Path(__file__).parents[1].joinpath('Data')

pd.options.display.expand_frame_repr = False

START_EPOCH = int(dt.datetime(2015, 6, 16).timestamp())  # day Trump announced candidacy
END_EPOCH = int(dt.datetime(2016, 11, 22).timestamp())  # 2 weeks after election day
UTC_TZ = tz.gettz('UTC')
NYC_TZ = tz.gettz('America/New_York')

API = PushshiftAPI()


def extract_author_metadata(submission_or_comment):
    """ Extracts author metadata from submission or comment object and returns a dict """
    if not hasattr(submission_or_comment, 'author_created_utc'):
        return None
    return {'author': submission_or_comment.author,
            'created': dt.datetime.fromtimestamp(submission_or_comment.author_created_utc, tz=UTC_TZ)}


def submission2dict(submission):
    """ Extracts certain attributes from submission object and returns a dict """
    data_dict = {'created': dt.datetime.fromtimestamp(submission.created_utc, tz=UTC_TZ)}
    attributes = ['domain', 'full_link', 'locked', 'selftext', 'subreddit', 'title', 'url']
    for a in attributes:
        try:
            data_dict[a] = getattr(submission, a)
        except AttributeError:
            data_dict[a] = None
    return data_dict


def comment2dict(comment):
    """ Extracts certain attributes from comment object and returns a dict """
    data_dict = {'created': dt.datetime.fromtimestamp(comment.created_utc, tz=UTC_TZ)}
    attributes = ['body', 'nest_level', 'parent_id', 'reply_delay', 'subreddit', 'id']
    for a in attributes:
        try:
            data_dict[a] = getattr(comment, a)
        except AttributeError:
            data_dict[a] = None
    return data_dict


def generate_submission_df(username, dirname):
    """ Generates a DataFrame of submissions from a given username """
    sub_dicts = []
    submissions = API.search_submissions(author=username, after=START_EPOCH, before=END_EPOCH)
    for s in submissions:
        sub_dicts.append(submission2dict(s))
    if not sub_dicts:
        return  # no submissions
    df = pd.DataFrame(data=sub_dicts)
    filepath = DATAPATH.joinpath(dirname, username)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    df.to_csv(filepath.joinpath('submissions.csv'))


def generate_comment_df(username, dirname):
    """ Generates a DataFrame of comments from a given username """
    comm_dicts = []
    comments = API.search_comments(author=username, after=START_EPOCH, before=END_EPOCH)
    for c in comments:
        comm_dicts.append(comment2dict(c))
    if not comm_dicts:
        return  # no comments
    df = pd.DataFrame(data=comm_dicts)
    filepath = DATAPATH.joinpath(dirname, username)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    df.to_csv(filepath.joinpath('comments.csv'))


def generate_suspicious_author_df(usernames):
    """ Generates a DataFrame of author metadata from a list of usernames """
    metadata_dicts = []
    for u in usernames:
        print('Fetching user: {}'.format(u))
        # just query any submission to get the metadata
        sub = API.search_submissions(author=u, limit=1, after=START_EPOCH, before=END_EPOCH)
        try:
            metadata_dicts.append(extract_author_metadata(submission_or_comment=next(sub)))
        except StopIteration:
            # no submissions - try looking for a comment
            try:
                comm = API.search_comments(author=u, limit=1, after=START_EPOCH, before=END_EPOCH)
                metadata_dicts.append(extract_author_metadata(submission_or_comment=next(comm)))
            except StopIteration:
                print('No comments or submissions in election period')
        sleep(0.1)  # avoid hitting API limits
    df = pd.DataFrame(data=metadata_dicts).set_index('author')
    df.to_csv(DATAPATH.joinpath('suspicious_accounts.csv'))


def generate_user_post_database(usernames, dirname):
    """ Gets comments and submissions for each user in list of usernames """
    for i, u in enumerate(usernames):
        print('Querying user {} - name: {}'.format(i, u))
        generate_submission_df(u, dirname)
        generate_comment_df(u, dirname)


def sample_normal_users(num_users):
    """
    Generates normal user csv.
    Users sampled by choosing a submission at a random time in the timeframe of interest
    """
    metadata_dicts = []
    already_sampled = set(
        [s.lower() for s in pd.read_csv(DATAPATH.joinpath('All_Accounts.csv'))['accountName'].values] +
        [s.lower() for s in pd.read_csv(DATAPATH.joinpath('normal_accounts2.csv'))['author'].values])
    try:
        for i in range(num_users):
            start_time = random_date(dt.datetime.utcfromtimestamp(START_EPOCH), dt.datetime.utcfromtimestamp(END_EPOCH))
            end_time = start_time + dt.timedelta(minutes=5)
            submission = next(API.search_submissions(limit=1,
                                                     after=int(start_time.timestamp()),
                                                     before=int(end_time.timestamp())))
            metadata = extract_author_metadata(submission)
            if metadata is None:
                continue
            u = metadata['author'].lower()
            if 'bot' not in u and 'auto' not in u and 'network' not in u and u not in already_sampled:
                metadata_dicts.append(metadata)
                print(f'Saving user {i}: {metadata["author"]}')
            sleep(0.1)  # avoid hitting API limits
    except Exception as e:
        print(e)
    df = pd.DataFrame(data=metadata_dicts).set_index('author')
    df.to_csv(DATAPATH.joinpath('normal_accounts3.csv'))


def get_user_posts(username):

    # search comments
    subs = API.search_submissions(author=username, limit=100, after=START_EPOCH, before=END_EPOCH)
    comms = API.search_comments(author=username, limit=100, after=START_EPOCH, before=END_EPOCH)

    # print results
    for r in subs:
        print(r.title)
    for r in comms:
        print(r.body)
        print(dir(r))
        print(r.id)


def add_post_numbers(filename):
    df = pd.read_csv(DATAPATH.joinpath(filename))
    users = df['author'].values
    num_subs, num_comms = [], []
    for u in users:
        subs, comms = load_user_posts(u)
        num_subs.append(0 if subs is None else len(subs))
        num_comms.append(0 if comms is None else len(comms))

    df['submissions'] = num_subs
    df['comments'] = num_comms
    print(df)
    df.to_csv(DATAPATH.joinpath(filename[:-4] + 'new' + '.csv'))


def moving_sum(time_hist, window_size):
    window_sums = []
    window_mid_points = []
    half_window = int(window_size / 2)
    for i, post_num in enumerate(time_hist):
        if i < 24 - window_size:
            window_sums.append(sum(time_hist[i:i + window_size]))
            window_mid_points.append(i + half_window)
        else:
            break
    min_window_sum = min(window_sums)
    min_midpoints = []
    for i, window_sum in enumerate(window_sums):
        if window_sum == min_window_sum:
            min_midpoints.append(window_mid_points[i])
    if len(min_midpoints) > 1:
        if window_size < 23:
            # recurse
            return moving_sum(time_hist, window_size + 2)
        else:
            return None
    else:
        return min_midpoints[0]


def calc_daily_downtime(time_hist):
    """
    Use a sliding window to find area with least posts.
    Args:
        time_hist: list of length 24, each idx corresponding to an hour of the day, containing post numbers

    Returns: single hour denoting middle of downtime (int) or None if downtime cannot be calculated
    """
    # we exclude users where less than 25% of the day has data, as this will not provide a clear downtime estimate
    min_posttime_thresh = 6
    if len([post_num for post_num in time_hist if post_num > 0]) < min_posttime_thresh:
        return None
    else:
        return moving_sum(time_hist, window_size=5)


def add_daily_downtime(filename):
    df = pd.read_csv(DATAPATH.joinpath(filename), index_col=0)
    try:
        users = df['author'].values
    except KeyError:
        users = df['accountName'].values
    downtimes = []
    for u in users:
        subs, comms = load_user_posts(u)
        posttimes = []
        posttimes += list(subs['created'].values) if subs is not None else []
        posttimes += list(comms['created'].values) if comms is not None else []
        user_posttimes = []
        user_downtime = None
        for ts in posttimes:
            try:
                parsed_ts = [int(ts.split()[1][:2])]
            except Exception as e:
                print(e)
                parsed_ts = []
            user_posttimes += parsed_ts

        try:
            hist = list(np.histogram(user_posttimes, bins=range(25))[0])
            user_downtime = calc_daily_downtime(hist)
        except Exception as e:
            print(e)
        downtimes.append(user_downtime)

    col = pd.Series(downtimes)
    col = col.fillna(col.mean())
    df['daily_downtime'] = col
    return df


def remove_banned_users(filename):
    df = pd.read_csv(DATAPATH.joinpath(filename), index_col=0)
    new_df = df.copy(deep=True)
    banned_users = []
    r = praw.Reddit(client_id='H8zTMC4ffxZiIg',
                     client_secret='xRf5E5UR6K3EjTnQ1HdS4jGNXSs',
                     user_agent='checkbanned by u/_____________l')
    for user in df['author'].values:
        try:
            if getattr(r.redditor(user), 'is_suspended', False):
                # account is suspended
                banned_users.append(user)
                new_df = new_df.loc[new_df['author'] != user, :]
        except NotFound:
            # account is shadowbanned or deleted
            banned_users.append(user)
            new_df = new_df.loc[new_df['author'] != user, :]

    print(len(banned_users))
    return new_df


def aggregate_normal_users(filenames):
    dfs = []
    for filename in filenames:
        dfs.append(remove_banned_users(filename))
    agg_df = pd.concat(dfs)
    agg_df.to_csv(DATAPATH.joinpath('normal_accounts_agg.csv'))


if __name__ == '__main__':
    # get_user_posts('BlackToLive')
    # generate_user_post_database(suspicious_account_usernames_with_posts)
    # sample_normal_users(1000)
    # users = pd.read_csv(DATAPATH.joinpath('normal_accounts3.csv'))['author'].values
    # print(users)
    # generate_user_post_database(users, 'NormalAccounts2')
    # add_post_numbers('suspicious_accounts.csv')
    # df = add_daily_downtime('All_Accounts_with_w2v.csv')
    # df.to_csv('All_Accounts_with_w2v_and_dd.csv')
    # add_post_numbers('normal_accounts.csv')
    # remove_banned_users('normal_accounts.csv')
    aggregate_normal_users(['normal_accounts.csv', 'normal_accounts2.csv'])
    # pass

