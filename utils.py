from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def adf_test(y):
    """
    Применяет критерий Дикки-Фуллера к временному ряду.
    """
    stat = adfuller(y)
    print('ADF Statistic:', stat[0])
    print('p-value:', stat[1])


def plot_acf_pacf(y, acf_title, pacf_title):
    """
    Строит графики ACF и PACF для временного ряда.
    """
    plt.figure(figsize=(10, 6), dpi=500)
    plot_acf(y)
    plt.xlabel('Лаг')
    plt.ylabel('ACF')
    plt.title(acf_title)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=500)
    plot_pacf(y)
    plt.xlabel('Лаг')
    plt.ylabel('PACF')
    plt.title(pacf_title)
    plt.show()
