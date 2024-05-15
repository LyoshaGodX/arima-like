import statsmodels.api as sm
from matplotlib import pyplot as plt
from visualization import plot_timeseries


def arima_model(y, x, order, title):
    """
    Обучает и строит прогноз для модели ARIMA.
    """
    model = sm.tsa.ARIMA(y, order=order)
    model_fit = model.fit()
    aic = model_fit.aic

    forecast = model_fit.predict(start=0, end=len(y) - 1)

    print(f"{title}")
    print(f"p, d, q: {order}")
    print(f"AIC тест: {aic}")
    print("------------------------------")

    plot_timeseries(x, y, title, "Проценты")
    plt.plot(range(0, len(y)), forecast, color='red')
    plt.legend(['Проценты', 'Прогноз'])
    plt.show()


def sarima_model(y, x, order, seasonal_order, title, plot_residuals=False):
    """
    Обучает и строит прогноз для модели SARIMA.
    Возвращает обученную модель.
    """
    model = sm.tsa.SARIMAX(y, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(maxiter=50)
    aic = model_fit.aic

    forecast = model_fit.predict(start=24, end=len(y) - 1)
    residuals = model_fit.resid

    print(f"{title}")
    print(f"p, d, q, P, D, Q, S: {order + seasonal_order}")
    print(f"AIC тест: {aic}")
    print("------------------------------")

    plot_timeseries(x, y, title, "Реальные данные")
    plt.plot(range(24, len(y)), forecast)
    plt.legend(['Реальные данные', 'Прогноз'])
    plt.show()

    if plot_residuals:
        return model_fit, residuals
    else:
        return model_fit