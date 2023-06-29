import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from scipy.stats import boxcox
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

# Чтение данных из файла Excel
df = pd.read_excel("Data.xlsx", index_col=0)

# Преобразование данных в словарь
data = df.to_dict()

x = []
y = []

january_2008_value = None

for year, values in data.items():
    for month, value in values.items():
        if year > 1997:
            x.append(f"{year}-{month}")
            if year == 2008 and month == 1:
                january_2008_value = value
            if year < 1998:
                y.append(value * 1000)
                continue
            y.append(value)

if january_2008_value is not None:
    # Преобразование значений в проценты от января 2008 года
    y = [(val / january_2008_value) * 100 for val in y]

# Удаление пропусков
y = np.array(y)
y = y[~np.isnan(y)]

x = np.arange(len(y))

# Создание меток для годов
labels = []
for i in range(1998, 2024):
    labels.append(f'{i}')

# График исходных данных
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x, y)
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title('Относительная номинальная заработная плата от 2008 года')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(['Проценты'])

# Критерий Дика-Фуллера
stat = adfuller(y)

# Вывод результатов
print('Исходный временной ряд')
print('ADF Statistic:', stat[0])
print('p-value:', stat[1])
print('------------------------------')

# Применение преобразования Бокса-Кокса
y_boxcox, lmda = boxcox(y)

# Критерий Дика-Фуллера
stat = adfuller(y_boxcox)

# Вывод результатов
print('Преобразованный ряд с помощью Бокса-Кокса')
print('ADF Statistic:', stat[0])
print('p-value:', stat[1])
print("Lambda:", lmda)
print('------------------------------')

# График Бокса-Кокса
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x, y_boxcox)
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title('Преобразование Бокса-Кокса')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(['Проценты'])

# Сезонное дифференцирование
y_seasonal = pd.Series(y_boxcox).diff(12).dropna().values
x_seasonal = np.arange(len(y_seasonal))

# Критерий Дика-Фуллера
stat = adfuller(y_seasonal)

# Вывод результатов
print('Сезонное дифференцирование, период 12 месяцев')
print('ADF Statistic:', stat[0])
print('p-value:', stat[1])
print('------------------------------')

# График после сезонного дифференцирования
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x_seasonal, y_seasonal)
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title('Первое сезонное дифференцирование')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(['Проценты'])
plt.show()

# Еще одно сезонное дифференцирование
y_seasonal_2 = pd.Series(y_seasonal).diff(12).dropna().values
x_seasonal_2 = np.arange(len(y_seasonal_2))
# Критерий Дика-Фуллера
stat = adfuller(y_seasonal_2)

# Вывод результатов
print('Второе сезонное дифференцирование, период 12 месяцев')
print('ADF Statistic:', stat[0])
print('p-value:', stat[1])
print('------------------------------')

# График после сезонного дифференцирования
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x_seasonal_2, y_seasonal_2)
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title('Второе сезонное дифференцирование')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(['Проценты'])
plt.show()

# Построение графика ACF для трансформированного ряда
plt.figure(figsize=(10, 6), dpi=500)
plot_acf(y_seasonal_2)
plt.xlabel('Лаг')
plt.ylabel('ACF')
plt.title('Каррелограмма ACF для трансформированного ряда')
plt.show()

# Построение графика PACF для трансформированного ряда
plt.figure(figsize=(10, 6), dpi=500)
plot_pacf(y_seasonal_2)
plt.xlabel('Лаг')
plt.ylabel('PACF')
plt.title('Каррелограмма PACF для трансформированного ряда')
plt.show()

# Модель ARIMA на аналитических параметрах
model_arima_analytical = sm.tsa.ARIMA(y_seasonal_2, order=(10, 0, 5))
model_arima_analytical_fit = model_arima_analytical.fit()
ARIMA_aic_analytical = model_arima_analytical_fit.aic

# Прогнозирование
forecast_arima_analytical = model_arima_analytical_fit.predict(start=0, end=len(y_seasonal_2) - 1)

# График модели ARIMA на аналитических параметрах
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x_seasonal_2, y_seasonal_2)
plt.plot(range(0, len(y_seasonal_2)), forecast_arima_analytical, color='red')
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title('Модель ARIMA на транформированном ряде, аналитические параметры')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(['Проценты', 'Прогноз'])
plt.show()

print('ARIMA, трансформированные данные, аналитические параметры')
print('p, d, q: 10, 0, 5')
print(f'AIC тест: {ARIMA_aic_analytical}')
print('------------------------------')

# Модель SARIMA
model_sarima_analytical = sm.tsa.SARIMAX(y, order=(10, 0, 5), seasonal_order=(2, 2, 1, 12))
model_sarima_analytical_fit = model_sarima_analytical.fit()
SARIMA_aic_analytical = model_sarima_analytical_fit.aic

# Прогнозирование
predictions_sarima_analytical = model_sarima_analytical_fit.predict(start=24, end=len(y) -1)

# График модели SARIMA
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x, y)
plt.plot(range(24, len(y)), predictions_sarima_analytical)
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title('Модель SARIMA на исходных данных, аналитические параметры')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(['Реальные данные', 'Прогноз'])
plt.show()

print('SARIMA, исходные данные, аналитические параметры')
print('p, d, q, P, D, Q, S: 10, 0, 5, 2, 2, 1, 12')
print(f'AIC тест: {SARIMA_aic_analytical}')
print('------------------------------')

# Автоподбор параметров SARIMA
model_sarima_auto = auto_arima(y, seasonal=True)
model_sarima_auto_fit = model_sarima_auto.fit(y)
SARIMA_aic_auto = model_sarima_auto_fit.aic()

print('SARIMA, исходные данные, автоматические параметры')
print('p, d, q, P, D, Q, S: 2, 1, 4, 0, 0, 0, 0')
print(f'AIC тест: {SARIMA_aic_auto}')
print('------------------------------')

# Остатки модели SARIMA на исходных данных
model_sarima_analytical_residuals = model_sarima_analytical_fit.resid

# График остатков
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x, model_sarima_analytical_residuals)
plt.xlabel('Год')
plt.ylabel('Остатки')
plt.title('Остатки SARIMA, исходные данные, аналитические параметры')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# Вывод суммы остатков в консоль
print('Сумма остатков SARIMA, аналитические параметры', np.sum(model_sarima_analytical_residuals))
print('------------------------------')

# Линейная регрессия на исходных данных
x = sm.add_constant(x)  # Регрессия на время
model = sm.OLS(y, x)
results = model.fit()
y_pred = results.predict(x)

# График линейной регрессии
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x[:, 1], y, label='Исходные данные')
plt.plot(x[:, 1], y_pred, label='Линейная регрессия')
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title('Регрессия на время, линейная')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend()
plt.show()

# Остатки от линейной регрессии
residuals = y - y_pred

# График остатков от линейной регрессии
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x[:, 1], residuals)
plt.xlabel('Год')
plt.ylabel('Остатки')
plt.title('Остатки линейной регрессии на время')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# Сумма остатков линейной регрессии
residual_sum = np.sum(residuals)
print('Сумма остатков линейной регрессии на время:', residual_sum)
print('------------------------------')

# Полиноминальная регрессия на исходных данных
degree = 2  # Степень многочлена
x_poly = sm.add_constant(x)
x_poly = sm.add_constant(x_poly) # Регрессия на время
for i in range(2, degree + 1):
    x_poly = np.column_stack((x_poly, x ** i))

poly_model = sm.OLS(y, x_poly)
poly_model_fit = poly_model.fit()

y_poly_pred = poly_model_fit.predict(x_poly)

# График полиноминальной регрессии
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x[:, 1], y, label='Исходные данные')
plt.plot(x[:, 1], y_poly_pred, label='Полиноминальная регрессия')
plt.xlabel('Год')
plt.ylabel('Относительная номинальная заработная плата')
plt.title(f'Полиноминальная регрессия на время (Степень: {degree})')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.legend()
plt.show()

# Остатки полиноминальной регрессии
poly_residuals = y - y_poly_pred

plt.figure(figsize=(10, 6), dpi=500)
plt.plot(x[:, 1], poly_residuals)
plt.xlabel('Год')
plt.ylabel('Остатки')
plt.title('Остатки полиноминальной регрессии на время')
plt.xticks(range(0, len(x), 12), labels, rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# Сумма остатков полиноминальной регрессии
poly_residual_sum = np.sum(poly_residuals)
print('Сумма полиноминальной регрессии на время:', poly_residual_sum)
print('------------------------------')
