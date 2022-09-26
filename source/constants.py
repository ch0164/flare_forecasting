################################################################################
# Filename: constants.py
# Description: This file contains constant variables common to each experiment.
################################################################################

import os

# ------------------------------------------------------------------------------
# --- Directories ---
RESULTS_DIRECTORY = os.getcwd() + "/"
SOURCE_DIRECTORY = os.path.abspath(os.path.join(RESULTS_DIRECTORY,
                                                os.pardir)) + "/"
ROOT_DIRECTORY = os.path.abspath(os.path.join(SOURCE_DIRECTORY,
                                              os.pardir)) + "/"
FLARE_LIST_DIRECTORY = ROOT_DIRECTORY + "flare_list/"
COINCIDENCE_LIST_DIRECTORY = FLARE_LIST_DIRECTORY + "coincident/"

FLARE_DATA_DIRECTORY = ROOT_DIRECTORY + "flare_data/"
FLARE_MEANS_DIRECTORY = FLARE_DATA_DIRECTORY + "time_series_means/"

RESULTS_DIRECTORY = ROOT_DIRECTORY + "results/"

CLEANED_DATA = "cleaned_data"

METRICS = "metrics"
FIGURES = "figures"
OTHER = "other"

# ------------------------------------------------------------------------------
# --- Flare Constants ---
# FLARE_CLASSES = ["NULL"]  # Used to test experiment faster
# FLARE_COLORS = ["grey"]   # Used to test experiment faster
FLARE_CLASSES = ["NULL", "B", "MX"]
FLARE_COLORS = ["grey", "dodgerblue", "orangered"]
COINCIDENCES = ["all", "coincident", "noncoincident"]

# ------------------------------------------------------------------------------
# --- Dataframe Headers ---
FLARE_PROPERTIES = [
    'ABSNJZH',
    'AREA_ACR',
    'MEANGAM',
    'MEANGBH',
    'MEANGBT',
    'MEANGBZ',
    'MEANJZD',
    'MEANJZH',
    'MEANPOT',
    'MEANSHR',
    'R_VALUE',
    'SAVNCPP',
    'SHRGT45',
    'TOTPOT',
    'TOTUSJH',
    'TOTUSJZ',
    'USFLUX',
    'd_l_f',
    'g_s',
    'slf',
]

LLA_HEADERS = [
    'LATMIN',
    'LATMAX',
    'LONMIN',
    'LONMAX',
]

# ------------------------------------------------------------------------------
# --- Plot Constants ---
LANDSCAPE_FIGSIZE = (28, 22)
LANDSCAPE_TITLE_FONTSIZE = 25
