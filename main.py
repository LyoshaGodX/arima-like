from matplotlib import pyplot as plt
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
