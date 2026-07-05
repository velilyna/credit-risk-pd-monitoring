# Credit Risk PD Modeling

Кодовый проект для сравнения моделей вероятности дефолта, проверки калибровки,
бизнес-порогов, устойчивости, статистической значимости различий и
интерпретации через SHAP.

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

## Структура

```text
credit-risk-pd-v3/
├── data/
│   └── default.xls
├── scripts/
│   └── run_pipeline.py
├── src/credit_risk/
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   ├── metrics.py
│   ├── statistics.py
│   ├── monitoring.py
│   ├── explainability.py
│   └── plots.py
├── reports/
│   └── figures/
├── artifacts/
├── requirements.txt
└── README.md
```

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

Ускоренный запуск без SHAP и bootstrap:

```bash
python scripts/run_pipeline.py --skip-shap --bootstrap-iterations 0
```

## Основные параметры

```bash
python scripts/run_pipeline.py \
  --fn-cost 100000 \
  --fp-cost 10000 \
  --bootstrap-iterations 2000 \
  --test-size 0.20 \
  --validation-size 0.20
```

## Выходные файлы

После запуска появятся:

- `reports/model_metrics.csv`
- `reports/model_comparisons_auc.csv`
- `reports/model_comparisons_cost.csv`
- `reports/feature_engineering_experiment.csv`
- `reports/feature_psi.csv`
- `reports/stress_metrics.csv`
- `reports/permutation_importance.csv`
- `reports/summary.json`
- графики в `reports/figures/`
- сохраненные модели в `artifacts/`

Отчет проект не генерирует: выводы можно написать вручную после анализа результатов.
