# Credit Risk PD Modeling

Кодовый проект для сравнения моделей вероятности дефолта, проверки калибровки,
бизнес-порогов, устойчивости, статистической значимости различий и
интерпретации через SHAP

## Что входит

- Logistic Regression
- Random Forest
- HistGradientBoosting
- LightGBM
- калибровка вероятностей
- ROC-AUC, Gini, KS, Average Precision, Brier Score
- выбор порога по стоимости ошибок
- paired bootstrap для разницы ROC-AUC
- paired bootstrap для разницы business cost
- PSI по признакам и по score
- искусственный stress scenario
- feature engineering experiment:
  - delinquency_momentum
  - payment_coverage_ratio
  - weighted_payment_shortfall
- permutation importance
- SHAP summary / bar / waterfall
- сохранение моделей и CSV-результатов

## Запуск

1. Положите исходный файл в `data/default.xls`.
2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Запустите:

```bash
python scripts/run_pipeline.py
```
