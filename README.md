# medical predistion mode 4th year project phase 1

This project trains and evaluates models to detect anxiety based on symptom features.

## Dataset

- Location: `/home/hertz/4thP/DataSet/Disease and symptoms dataset (Copy).csv`
- Target label: `diseases == "anxiety"` (binary classification)

## Models and Artifacts

- Trained model: `/home/hertz/4thP/backend/models/anxiety_model.pkl`
- Feature columns: `/home/hertz/4thP/backend/models/feature_columns.pkl`

## Training and Evaluation

### Logistic Regression

Run:

```bash
python phase1.py
```

### Decision Tree

Run:

```bash
python phase1-1.py
```

### Train and Save Model Artifacts

Run:

```bash
python helpers/model.py
```

This saves model files to `/home/hertz/4thP/backend/models`.

## API Inference

Start the FastAPI server:

```bash
python backend/main.py
```

The `/predict` endpoint accepts either:

- `{"symptoms": ["symptom name", "symptom name"]}`
- A raw symptom dictionary with `0/1` values

## Notes

- Update paths in scripts if you move the dataset or models.
- Install dependencies from `requirements.txt` before running.
