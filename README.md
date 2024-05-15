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
![seasonal_diff_1.png](images%2Fseasonal_diff_1.png)![Seasonal_diff1.png](images%2FSeasonal_diff.png)
------------------------------
### Второе сезонное дифференцирование, период 12 месяцев
- ADF Statistic: `-3.2848675417477553`
- p-value: `0.015568595821135255`
![seasonal_diff_2.png](images%2Fseasonal_diff_2.png)
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
![SARIMA_remains.png](images%2FSARIMA_remains.png)
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

## Вывод
Модель SARIMA показала наилучший результат по сравнению с другими моделями.
