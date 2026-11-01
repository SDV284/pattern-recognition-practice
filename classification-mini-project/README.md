# Порівняння класичних моделей класифікації

Мініпроєкт практичних робіт 4–7 з дисципліни «Розпізнавання образів та класичне машинне навчання».

**Студент:** Святішенко Дмитро, група ІН-32/2

> Результати навчального мініпроєкту не є медичним діагнозом і не повинні використовуватися для клінічних рішень.

## Постановка задачі

За характеристиками пацієнта визначити, чи належить він до класу з наявністю ознак серцевого захворювання.
Задача — бінарна класифікація з навчанням за розміченими даними.

- цільова змінна: `target` (0 — ознаки захворювання відсутні, 1 — наявні);
- **позитивний клас: 1**;
- основна метрика (обрана до експериментів): **F1 позитивного класу**; додаткові — ROC AUC і recall.

## Набір даних

[Heart Disease, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) — файл
`processed.cleveland.data`, збережений як [`data/heart_disease.csv`](data/heart_disease.csv).

- 303 об'єкти, 13 вхідних ознак, 6 пропусків (`ca` — 4, `thal` — 2);
- числові ознаки: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`;
- категоріальні ознаки (числові коди): `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`;
- ціль `num` = 0 → `target` = 0, `num` = 1–4 → `target` = 1; класи 164 / 139.

## Методика експерименту

- фіксований стратифікований train/test split 80/20 (`random_state=42`): 242 / 61 об'єкт;
- підбір алгоритму й гіперпараметрів лише на train;
- єдина `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` для всіх моделей;
- правило вибору: найвищий CV F1 → короткий список (відставання ≤ 0,01) → менший std і більший recall → простіша модель;
- фінальну модель зафіксовано до відкриття test; одноразове фінальне оцінювання на test.

## Підготовка даних

- `SimpleImputer` (медіана для числових, мода для категоріальних ознак);
- `StandardScaler` для числових ознак;
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` для категоріальних;
- `ColumnTransformer`;
- `Pipeline` — усі перетворення навчаються лише на train-частині кожного фолду.

## Досліджені моделі

- Logistic Regression — масштабування, L1/L2, C = 0,01…100;
- k-NN — масштабування, k = 3…15, евклідова / мангеттенська відстань;
- Gaussian Naive Bayes;
- LDA;
- QDA — стабільність і `reg_param`;
- SVM — linear і RBF, `GridSearchCV` за C і gamma.

## Порівняння за крос-валідацією

| Модель | Найкращі параметри | CV F1 | CV F1 std | CV Recall | CV ROC AUC |
|---|---|---:|---:|---:|---:|
| **Logistic Regression** | L2, C = 10 | **0,836** | 0,013 | 0,802 | **0,900** |
| SVM | RBF, C = 10, gamma = 0,01 | 0,827 | 0,033 | 0,783 | 0,897 |
| k-NN | k = 3, мангеттенська | 0,818 | 0,030 | 0,792 | 0,877 |
| LDA | — | 0,813 | 0,026 | 0,766 | 0,899 |
| QDA | reg_param = 0,7 | 0,802 | 0,045 | 0,756 | 0,885 |
| GaussianNB | — | 0,749 | 0,021 | 0,846 | 0,877 |

Повна таблиця: [`reports/cv_model_comparison.csv`](reports/cv_model_comparison.csv).
QDA без регуляризації не навчається (вироджена коваріаційна матриця після one-hot-кодування).

## Фінальна модель і test-результат

**Логістична регресія, L2, C = 10** (solver liblinear) — найвищий CV F1 і CV ROC AUC, найменший розкид між фолдами,
найвищий recall у короткому списку, проста та інтерпретована.

| Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC AUC |
|---:|---:|---:|---:|---:|
| 0,869 | 0,833 | 0,893 | 0,862 | 0,962 |

Помилок 8 із 61 (FN = 3, FP = 5). Test F1 узгоджується з CV (0,836); вищі recall і ROC AUC пояснюються малим
розміром тестової вибірки. Після перегляду test модель не змінювалася.

Файли: [`reports/final_test_metrics.csv`](reports/final_test_metrics.csv),
[`models/best_classifier.joblib`](models/best_classifier.joblib) — увесь Pipeline разом із препроцесингом.

## Структура

```
classification-mini-project/
├── data/heart_disease.csv
├── notebooks/PR04_Sviatishenko_D.ipynb
├── models/best_classifier.joblib
├── reports/
│   ├── cv_model_comparison.csv
│   ├── final_test_metrics.csv
│   ├── PR04_07_control_questions.md
│   └── figures/
└── README.md
```

## Запуск

Відкрити [Notebook](notebooks/PR04_Sviatishenko_D.ipynb) у Google Colab і послідовно виконати всі комірки
(Runtime → Run all). Якщо файла `data/heart_disease.csv` немає, Notebook завантажить його з UCI.

```python
import joblib
model = joblib.load("models/best_classifier.joblib")
model.predict(new_patients)  # сирі ознаки, без ручного масштабування чи кодування
```

[Відповіді на контрольні запитання](reports/PR04_07_control_questions.md)
