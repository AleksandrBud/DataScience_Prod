import pandas as pd
from sklearn.metrics import classification_report


def get_classification_report(y, y_predict, name):
    print(f'{name}\n\n{classification_report(y, y_predict)}')

