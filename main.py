
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from data_visualization import process_data, perform_TSNE
# visulaize the important characteristics of the dataset
import matplotlib.pyplot as plt


def battles():

    filename = "battles.csv"
    labels = ["attacker_outcome", "major_death", "major_capture"]
    remove_columns = [0, 1, 2]
    for label in labels:
        x, y = process_data(filename, label, remove_columns, 0)
        perform_TSNE(x, y)


def charachter_deaths():
    filename = "character-deaths.csv"
    labels = ["Death Year", "Book of Death"]
    remove_columns = [0, 1, 3, 4]
    for label in labels:
        x, y = process_data(filename, label, remove_columns, 1)
        perform_TSNE(x, y)



def charachter_predictions():
    pass

battles()
charachter_deaths()
