import random
import json
import os
import warnings

from IPython.display import display, Audio
import librosa
import matplotlib.pyplot as plt
import mir_eval
import mir_eval.display
import numpy as np
from keras.layers import LSTM, Activation, Convolution1D, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence
import pretty_midi

warnings.filterwarnings(
    'ignore')  # TODO Remove when librosa.cqt has been updated.

DATA_PATH = '.'
RESULTS_PATH = '.'
SCORE_FILE = 'match_scores.json'


def msd_path(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


def mp3_path(msd_id):
    """Given an MSD ID, return the path to the corresponding MP3 file."""
    return os.path.join(DATA_PATH, 'lmd_matched_mp3',
                        msd_path(msd_id) + '.mp3')


def midi_path(msd_id, midi_md5, kind='aligned'):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file."""
    # TODO Enums kind: 'matched' or 'aligned'
    return os.path.join(RESULTS_PATH, 'lmd_{}'.format(kind), msd_path(msd_id),
                        midi_md5 + '.mid')