import pandas as pd
import numpy as np
from scipy.stats import boxcox


def read_data(file_path):
    """
    Читает данные из CSV-файла и возвращает DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Преобразует данные в словарь, создает массивы x и y.
    Возвращает словарь данных, массивы x и y.
    """
    data = df.set_index('Год').T.to_dict()  # Convert DataFrame to dictionary
    x, y = [], []
    january_2008_value = None

    for year, values in data.items():  # Iterate over the dictionary
        year = int(year)  # Convert year to integer
        for month, value in values.items():
            if year > 1997:
                x.append(f"{year}-{month}")
                if year == 2008 and month == '1':
                    january_2008_value = value
                if year < 1998:
                    y.append(value * 1000)
                    continue
                y.append(value)

    if january_2008_value is not None:
        y = [(val / january_2008_value) * 100 for val in y]

    y = np.array(y)

    # Remove corresponding x values where y is NaN
    x = [xi for xi, yi in zip(x, y) if not np.isnan(yi)]
    y = y[~np.isnan(y)]

    return data, x, y


def box_cox_transform(y):
    """
    Применяет преобразование Бокса-Кокса к временному ряду.
    Возвращает преобразованный ряд и значение лямбды.
    """
    y_boxcox, lmda = boxcox(y)
    return y_boxcox, lmda


def seasonal_differencing(y, period):
    """
    Применяет сезонное дифференцирование к временному ряду.
    Возвращает дифференцированный ряд.
    """
    y_seasonal = pd.Series(y).diff(period).dropna().values
    return y_seasonal
