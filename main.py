import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox, het_goldfeldquandt
from data_processing import read_data, preprocess_data, box_cox_transform, seasonal_differencing
from visualization import plot_timeseries, plot_residuals
from arima_modeling import arima_model, sarima_model
from regression import linear_regression, polynomial_regression
from utils import adf_test, plot_acf_pacf


def main():
    # Чтение данных
    df = read_data("Data.csv")

    # Предобработка данных
    data, x, y = preprocess_data(df)

    # Разделение данных на обучающую и тестовую выборки
    x_train, x_test = x[:-12], x[-12:]
    y_train, y_test = y[:-12], y[-12:]

    # Исходный временной ряд
    plot_timeseries(x, y, "Относительная номинальная заработная плата от 2008 года", "Проценты")
    print("Исходный временной ряд")
    adf_test(y)
    print("------------------------------")

    # Преобразование Бокса-Кокса
    y_boxcox, lmda = box_cox_transform(y)
    plot_timeseries(x, y_boxcox, "Преобразование Бокса-Кокса", "Проценты")
    print("Преобразованный ряд с помощью Бокса-Кокса")
    adf_test(y_boxcox)
    print(f"Lambda: {lmda}")
    print("------------------------------")

    # Сезонное дифференцирование
    y_seasonal = seasonal_differencing(y_boxcox, 12)
    x_seasonal = x[12:]
    plot_timeseries(x_seasonal, y_seasonal, "Первое сезонное дифференцирование", "Проценты")
    print("Сезонное дифференцирование, период 12 месяцев")
    adf_test(y_seasonal)
    print("------------------------------")

    # Второе сезонное дифференцирование
    y_seasonal_2 = seasonal_differencing(y_seasonal, 12)
    x_seasonal_2 = x_seasonal[12:]
    plot_timeseries(x_seasonal_2, y_seasonal_2, "Второе сезонное дифференцирование", "Проценты")
    print("Второе сезонное дифференцирование, период 12 месяцев")
    adf_test(y_seasonal_2)
    print("------------------------------")

    # Построение графиков ACF и PACF
    plot_acf_pacf(y_seasonal_2, "Каррелограмма ACF для трансформированного ряда",
                  "Каррелограмма PACF для трансформированного ряда")

    # Модель ARIMA
    arima_model(y_seasonal_2, x_seasonal_2, (10, 0, 5),
                "Модель ARIMA на трансформированном ряде")

    # Модель SARIMA
    sarima_model_fit, sarima_model_residuals = sarima_model(y_train, x_train, (10, 0, 5), (2, 2, 1, 12),
                                          "Модель SARIMA на обучающих данных", True)
    plot_residuals(x_train, sarima_model_residuals, "Остатки SARIMA")
    print(f"Сумма остатков SARIMA: {sum(sarima_model_residuals)}")
    print("------------------------------")

    # Тест Льюнга-Бокса
    lb_test = acorr_ljungbox(sarima_model_residuals, lags=10)
    lb_stat = lb_test['lb_stat'][1]
    p_value = lb_test['lb_pvalue'][1]

    if p_value < 0.05:
        print(f"lb-test = {lb_stat}, p = {p_value}. Не можем отвергнуть H0: остатки распределяются независимо")
    else:
        print(f"lb-test = {lb_stat}, p = {p_value}. Отвергаем H0: серия остатков не является белым шумом")

    # Тест Гольдфельда-Квандта
    x_train = pd.to_datetime(x_train)
    x_train = x_train.view(np.int64) // 10 ** 9
    X_train = sm.add_constant(x_train)  # Добавляем константу к вектору x_train
    gq_test = het_goldfeldquandt(sarima_model_residuals, X_train)

    if gq_test[1] < 0.05:
        print(f"F-stat = {gq_test[0]}, p = {gq_test[1]}. Не можем отвергнуть H0: остатки гомоскедастичны")
    else:
        print(f"F-stat = {gq_test[0]}, p = {gq_test[1]}. Отвергаем H0: остатки: остатки гетероскедастичны")

    # Прогнозирование на тестовых данных
    forecast = sarima_model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

    # Визуализация реальных данных и прогноза на тестовых данных
    plt.figure(figsize=(12, 6))
    plt.plot(x_test, y_test, label="Реальные данные")
    plt.plot(x_test, forecast, color='red', label="Прогноз")
    plt.title("Прогноз на тестовых данных")
    plt.xlabel("Time")
    plt.ylabel("Проценты")
    plt.grid(True)
    labels = [x_test[i] for i in range(len(x_test))]
    plt.xticks(range(len(x_test)), labels, rotation=45, fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Линейная регрессия
    linear_regression(x, y, "Регрессия на время, линейная")

    # Полиномиальная регрессия
    degree = 2
    polynomial_regression(x, y, degree, f"Полиноминальная регрессия на время (Степень: {degree})")


if __name__ == "__main__":
    main()
