import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt

from visualization import plot_timeseries, plot_residuals


def linear_regression(x, y, title):
    """
    Выполняет линейную регрессию на времени.
    """
    x = sm.add_constant(np.arange(len(y)))
    model = sm.OLS(y, x)
    results = model.fit()
    y_pred = results.predict(x)

    residuals = y - y_pred

    plot_timeseries(x[:, 1], y, title, "Исходные данные")
    plt.plot(x[:, 1], y_pred, label='Линейная регрессия')
    plt.legend()
    plt.show()

    plot_residuals(x[:, 1], residuals, "Остатки линейной регрессии на время")
    print(f"Сумма остатков линейной регрессии на время: {sum(residuals)}")
    print("------------------------------")


def polynomial_regression(x, y, degree, title):
    """
    Выполняет полиномиальную регрессию на времени.
    """
    x_poly = sm.add_constant(np.arange(len(y)))
    x_poly = sm.add_constant(x_poly)
    for i in range(2, degree + 1):
        x_poly = np.column_stack((x_poly, np.arange(len(y)) ** i))

    poly_model = sm.OLS(y, x_poly)
    poly_model_fit = poly_model.fit()

    y_poly_pred = poly_model_fit.predict(x_poly)
    poly_residuals = y - y_poly_pred

    plot_timeseries(x_poly[:, 1], y, title, "Исходные данные")
    plt.plot(x_poly[:, 1], y_poly_pred, label='Полиномиальная регрессия')
    plt.legend()
    plt.show()

    plot_residuals(x_poly[:, 1], poly_residuals, "Остатки полиномиальной регрессии на время")
    print(f"Сумма остатков полиномиальной регрессии на время: {sum(poly_residuals)}")
    print("------------------------------")
