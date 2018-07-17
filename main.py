
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
    filename = "character-predictions.csv"
    labels = ["isNoble", "isPopular", "isAlive"]
    remove_columns = [[0, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17], [0, 5, 7, 8, 9, 10, 11]]
    for count,label in enumerate(labels):
        x, y = process_data(filename, label, remove_columns[count], 0)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
        perform_TSNE(x_test, y_test)
battles()
charachter_deaths()
charachter_predictions()