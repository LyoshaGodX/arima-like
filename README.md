## Зависимости

Для работы с проектом вам потребуются следующие библиотеки:

- numpy
- pandas
- matplotlib
- statsmodels
- scipy

```bash
pip install numpy pandas matplotlib statsmodels scipy
```

### Исходный временной ряд
- ADF Statistic: `2.568174303850929`
- p-value: `0.999068587168079`
![Initial_data.png](images%2FInitial_data.png)
------------------------------
### Преобразованный ряд с помощью Бокса-Кокса
- ADF Statistic: `-1.6007991776114638`
- p-value: `0.48319200312423627`
- Lambda: `0.3843530845588325`
![Box_Cox.png](images%2FBox_Cox.png)
------------------------------
### Сезонное дифференцирование, период 12 месяцев
- ADF Statistic: `-1.9314540411024517`
- p-value: `0.31741184180934`
![seasonal_diff_1.png](images%2Fseasonal_diff_1.png)
------------------------------
### Второе сезонное дифференцирование, период 12 месяцев
- ADF Statistic: `-3.2848675417477553`
- p-value: `0.015568595821135255`
![seasonal_diff_2.png](images%2Fseasonal_diff_2.png)
------------------------------
### Коррелограммы
- Единственный сезонный лаг в области значимости автокорреляции – 12-й, таким образом, 
параметр `𝑷 = 𝟏`. Последний несезонный лаг, автокорреляция которого статистически 
значима – пятый. Таким образом, `𝒑 = 𝟓`

![ACF.png](images%2FACF.png)
- Последний значимый сезонный лаг – 24, таким образом, параметр `𝑸 = 𝟐`. Последний 
значимый несезонный лаг при этом 14-й, что больше периода сезонности (𝑆 = 12). А 
значит выберем последний значимый несезонный лаг, предшествующий 12-му, то есть 
10-й. Таким образом, `𝒒 = 𝟏𝟎`.

![PACF.png](images%2FPACF.png)
------------------------------
### Модель ARIMA на трансформированном ряде
- p, d, q: `(10, 0, 5)`
- AIC тест: `-120.97648830571066`
![ARIMA.png](images%2FARIMA.png)
------------------------------
### Модель SARIMA на обучающих данных
- p, d, q, P, D, Q, S: `(10, 0, 5, 2, 2, 1, 12)`
- AIC тест: `1563.9200751600397`
![SARIMA_learn.png](images%2FSARIMA_learn.png)
------------------------------
### Остатки SARIMA
- Сумма остатков SARIMA: `172.13576143392078`
- F-stat = `4.938280554395085`, p = `4.0522341789610836e-20`. **Не можем отвергнуть** H0: остатки гомоскедастичны.
- lb-test = `3.1734288489969664`, `p = 0.07484516308967297`. **Отвергаем** H0: серия остатков не является белым шумом
![SARIMA_remains.png](images%2FSARIMA_remains.png)

------------------------------
Тест Льюнга-Бокса и Гольдфельда-Квандта:
```py
lb_test = acorr_ljungbox(sarima_model_residuals, lags=10)
    lb_stat = lb_test['lb_stat'][1]
    p_value = lb_test['lb_pvalue'][1]

    if p_value < 0.05:
        print(f"lb-test = {lb_stat}, p = {p_value}. Не можем отвергнуть H0: остатки распределяются независимо")
    else:
        print(f"lb-test = {lb_stat}, p = {p_value}. Отвергаем H0: серия остатков не является белым шумом")

gq_test = het_goldfeldquandt(sarima_model_residuals, X_train)

    if gq_test[1] < 0.05:
        print(f"F-stat = {gq_test[0]}, p = {gq_test[1]}. Не можем отвергнуть H0: остатки гомоскедастичны")
    else:
        print(f"F-stat = {gq_test[0]}, p = {gq_test[1]}. Отвергаем H0: остатки: остатки гетероскедастичны")
```
- F-stat = `4.938280554395085`, p = `4.0522341789610836e-20`. Не можем отвергнуть H0: остатки гомоскедастичны. Дисперсия в остатках, если я ничего не напутал, постоянная, на глаз мне казалось, что преобразование Бокса-Кокса плохо сработало.
- lb-test = `3.1734288489969664`, p = '0.07484516308967297`. Отвергаем H0: серия остатков не является белым шумом. В остатках есть автокорреляция. А вот это плохо, можно посмотреть в сторону других моделей или изменить порядок авторегрессии
------------------------------
### Прогноз SARIMA
![SARIMA_test.png](images%2FSARIMA_test.png)
------------------------------
### Линейная регрессия
- Сумма остатков линейной регрессии на время: `2.4840574042173102e-11`
![linreg.png](images%2Flinreg.png)
![linreg_remains.png](images%2Flinreg_remains.png)
------------------------------
### Полиномиальная регрессия
- Сумма остатков полиномиальной регрессии на время: `1.388059445162071e-09`
![polyreg.png](images%2Fpolyreg.png)
![polyreg_remains.png](images%2Fpolyreg_remains.png)

