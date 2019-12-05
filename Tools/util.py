import os
import pandas as pd
from random import randrange
from datetime import timedelta
import pathlib

DATAPATH = pathlib.Path(__file__).parents[1].joinpath('Data')


def load_user_posts(username):
    """ Loads submission and comment DataFrames for a given user in our database """
    dirs = ['SuspiciousAccounts', 'NormalAccounts', 'NormalAccounts2']
    for d in dirs:
        filepath = DATAPATH.joinpath(d, username)
        if os.path.exists(filepath):
            break
    else:
        raise ValueError(f'No data for {username} in our database')
    try:
        subs = pd.read_csv(filepath.joinpath('submissions.csv'))
    except FileNotFoundError:
        subs = None
    try:
        comms = pd.read_csv(filepath.joinpath('comments.csv'))
    except FileNotFoundError:
        comms = None
    return subs, comms


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    From https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)
