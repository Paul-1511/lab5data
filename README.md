# Laboratorio 5: Clasificacion de tweets sobre desastres

CC3084 - Data Science - Universidad del Valle de Guatemala - Semestre II 2026

## Objetivo

Clasificar si un tweet se refiere a un desastre real (`target=1`) o no (`target=0`), utilizando tecnicas de mineria de texto.

## Dataset

[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) (Kaggle).

| Archivo | Filas | Descripcion |
|---------|-------|-------------|
| `train.csv` | 7,613 | Datos etiquetados (columnas: id, keyword, location, text, target) |
| `test.csv` | 3,263 | Datos sin etiqueta para prediccion final |
| `sample_submission.csv` | 3,263 | Plantilla de envio |

## Estructura del proyecto

```
lab5data/
  train.csv
  test.csv
  sample_submission.csv
  submission_lab5.csv       # Predicciones de test.csv (generado)
  lab5.ipynb                # Notebook principal
  analisis_lab5.py          # Script modular (CLI)
  md2pdf.py                 # Generador de PDF desde Markdown
  INFORME_LAB5.md           # Informe academico
  GUIA_REPLICACION_LAB5.md  # Tutorial de replicacion
  requirements.txt
  .gitignore
  outputs/
    eda/                    # Figuras EDA
      tables/               # Tablas CSV de n-gramas
      manifest.json
    models/                 # Artefactos de modelado
      cv_comparison.csv
      holdout_metrics.json
      confusion_matrix.png
      pr_roc_curves.png
      split_manifest.json
      best_pipeline.joblib
    sentiment/              # Artefactos de sentimiento
      sentiment_scores.csv
      ablation_results.json
      mann_whitney.json
```

## Uso

### Notebook

Abrir `lab5.ipynb` en Jupyter y ejecutar todas las celdas.

### Script desde terminal

```bash
pip install -r requirements.txt

# Fase 1: EDA
python analisis_lab5.py --eda
python analisis_lab5.py --validate-eda

# Fase 2: Modelos
python analisis_lab5.py --models
python analisis_lab5.py --predict-test
python analisis_lab5.py --validate-models

# Fase 3: Sentimiento y negatividad
python analisis_lab5.py --sentiment
python analisis_lab5.py --models-sentiment
python analisis_lab5.py --validate-sentiment

# PDF e informe
python md2pdf.py INFORME_LAB5.md INFORME_LAB5.pdf

# Validacion final
python analisis_lab5.py --validate-final
```

## Modelos comparados

| Modelo | Descripcion |
|--------|-------------|
| Dummy | Baseline: clase mayoritaria |
| LogReg | Regresion Logistica (balanced) |
| ComplementNB | Complement Naive Bayes |
| LinearSVC | SVM lineal calibrado |
| LogReg_char | LogReg con n-gramas de caracteres |

Seleccion por F1(target=1) con CV estratificado por grupos. Evaluacion final en holdout (~20%).

## Funcion de clasificacion

```python
from analisis_lab5 import clasificar_tweet
result = clasificar_tweet("Earthquake hits California, buildings collapsed")
# {'clase': 1, 'interpretacion': 'Desastre', 'score': 0.92, 'score_tipo': 'probabilidad'}
```

## Semilla

Todas las operaciones aleatorias usan `SEED = 42`.
