# IVF Journey Tracker

## Overview

An AI-assisted IVF journey tracking platform for recording treatment information, managing medical documents, monitoring wellness, exploring clinics, and providing IVF-related insights.

This repository contains the Python/Flask application currently present in the project root. It is intended for software development and research only, not for clinical diagnosis or treatment decisions.

## Main Features

- IVF success prediction and treatment analytics
- Medical report upload, PDF extraction, and OCR
- Emotional wellness and mindfulness tracking
- AI chatbot, nutrition, yoga, and wellness helpers
- IVF clinic directory and search
- Patient, doctor, and admin dashboards
- Medical document management and sharing
- Authentication, messaging, forum, notes, reminders, and notifications

## Technology Stack

- Python and Flask
- Flask-SQLAlchemy with SQLite by default, or a database URL supplied through `DATABASE_URL`
- SQLAlchemy ORM models in `models.py`
- NumPy, Pandas, scikit-learn, XGBoost, LightGBM, SHAP, PyTorch, and joblib for machine-learning work
- OpenAI and Google Gemini integrations through environment variables
- `pytesseract`, Pillow, and `pdfminer.six` for document processing/OCR
- HTML, CSS, and JavaScript frontend assets served by Flask

No Node.js `package.json` application is present in the project root.

## AI and ML

### Production application

The Flask application starts from `main.py`. Prediction services are provided by `prediction_service.py` and the model modules. Existing trained model files are kept separate from the offline experiment code and are not replaced by this repository preparation.

### Offline model comparison

The isolated experiment scripts compare IVF-success classifiers without importing or modifying Flask, the database, routes, or production model files. They use:

- Training data: `ivf_5000_dataset.xlsx`
- External synthetic validation data: `IVF_validation_dataset_500_synthetic.csv`
- Models and reports: generated comparison scripts and result files are local experiment artifacts and are ignored by default

The external validation dataset is synthetic and contains no real patient information. Results are research observations only. The external set should not be used as clinical evidence or as a replacement for the production model.

## Project Structure

```text
.
├── main.py
├── routes.py
├── database.py
├── models.py
├── prediction_service.py
├── medical_document_analyzer.py
├── openai_service.py
├── model_training.py
├── preprocessing.py
├── train_*.py
├── *models.py
├── templates and HTML pages
├── static/
├── models/
├── ml_models/
├── knowledge_base/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```

The development server is available at `http://127.0.0.1:5000`.

Do not commit `.env`, virtual environments, databases, uploaded documents, logs, or private credentials. Set real API keys only in the local environment when the related feature is needed.

## Database

SQLite is used by default through `sqlite:///ivf_tracker.db`; Flask-SQLAlchemy resolves this relative SQLite database under the Flask instance directory. A different SQLAlchemy-compatible URL can be supplied as `DATABASE_URL`. The project includes additive schema migration helpers for existing local databases. Never commit local database files or backups.

## Datasets and Documents

The repository includes development datasets and model-training assets where their size and provenance make that appropriate. Uploaded medical documents, local database files, generated reports, and OCR artifacts are excluded by `.gitignore`. Keep any private or identifiable data outside the repository.

## Security

Use `.env.example` as the configuration template. Real secrets, passwords, tokens, database credentials, certificates, and private keys must remain outside Git. Rotate any credential that may previously have been exposed.

## License

No license file is currently present. Add the project’s intended license before distributing the repository publicly.