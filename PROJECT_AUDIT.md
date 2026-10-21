# IVF Journey Tracker Project Audit

## 1. Scope and repository reality

This project is a Flask-based IVF journey tracker and prototype research app. The repository mixes:

- runtime application code in `main.py`, `models.py`, `database.py`, `prediction_service.py`, and `openai_service.py`
- ML experimentation code and training artifacts in `train_*.py`, `ivf_model_metadata.json`, `saved_models/`, and `ml_models/`
- dataset-generation and validation artifacts under `results/`, `data/`, and generated CSV/XLSX files
- frontend templates and static assets under `templates/` and `static/`
- an app-level test suite under `tests/`

The codebase contains a real working prototype, but it also contains multiple duplicate implementations, stale experiment artifacts, and several mismatches between the research narrative and the actual runtime code.

## 2. Current architecture

### Application layer

- `main.py` is the primary Flask application entry point.
- `models.py` defines the ORM models for users, patient data, IVF cycles, wellness logs, predictions, documents, messages, notifications, and clinics.
- `database.py` initializes the SQLAlchemy database object.
- `schema_migration.py` and `migrate_*.py` scripts handle schema repair and migration tasks.

### AI and prediction layer

- `prediction_service.py` wraps prediction logic and rule-based fallbacks.
- `ml_models/` contains model-related training and preprocessing utilities.
- `saved_models/` and `ivf_success_model.pkl` provide persisted model artifacts.
- `openai_service.py` contains a fallback IVF chatbot and optional external API integration.

### Medical document workflow

- `medical_document_workflow.py` defines OCR mapping, validation checks, and prediction-field validation logic.
- `main.py` implements upload handling, hashing, duplicate detection, report freshness checks, and dashboard-feature reconciliation.

### Data and evaluation artifacts

- `generate_ivf_5000.py` and related scripts generate synthetic IVF records.
- `ivf_model_metadata.json` and `results/validation/*` together document the synthetic provenance and validation approach.
- There are multiple comparison scripts and report artifacts that appear to be research outputs rather than core app runtime.

## 3. Working features confirmed by code and tests

These features are present and have been validated in the repository’s current state:

- Flask app bootstrapping and page rendering for core web routes
- user creation, password hashing, and session-based auth flow
- patient profile persistence and relationship storage in SQLAlchemy
- IVF predictor page and server-side prediction pathways
- wellness logging and dashboard-style data aggregation
- document upload, hashing, duplicate detection, and validation workflow
- prediction payload validation and protection against missing/invalid values
- API and template support for chatbot, FAQ, clinic search, and dashboard-related views

## 4. Incomplete or unstable features

These were confirmed as incomplete, stale, or fragile:

- package import stability: the `ml_models` package referenced an import path that does not exist (`ml_models_manager`)
- test discovery contamination: nested unrelated project folders caused pytest collection collisions from other repos in the workspace
- stale or duplicate project structures: there are nested directories such as `ChoreCompanion/` and `Preditive-IVF-analytics-system-project/` that create import-path and collection confusion
- model consistency drift: the repo contains both RandomForest and XGBoost references without a single canonical runtime contract
- document workflow and route assumptions are inconsistent with a clean login/session state, leading to route crashes when session data is absent
- model persistence and scientific provenance are not fully coherent across runtime, training, and validation artefacts

## 5. Confirmed bugs and security issues

### Confirmed bugs

- `models.py` originally exposed runtime incompatibilities: invalid `WellnessLog` constructor usage, relationship behavior mismatch, and missing compatibility fields
- route handlers assumed session state existed without checking it, which caused crashes for unauthenticated or stale sessions
- nested test folders created pytest import and path conflicts
- some status values and model field mappings were not consistently aligned with the actual `PREDICTION_FIELD_MAP`

### Security and privacy concerns

- the app uses a default development secret key (`SESSION_SECRET` fallback) and should not be treated as production-safe
- AI and chatbot features can expose fertility-health guidance without a clear risk disclaimer enforced at runtime
- uploads and documents are stored locally in the workspace and need stronger access-control and sanitization review
- model outputs are presented as if they are clinically useful without a strict “research-only, not clinical” guardrail in all user-facing messaging
- all demonstration users and credentials are seeded in dev mode and should never be left active in production

## 6. Missing tests and test gaps

The project currently includes tests for some routes and document workflows, but there are gaps in coverage for:

- production-safe session/auth edge cases
- privacy and document access control
- model provenance and data-leakage checks
- security boundary validation for uploaded documents and AI responses
- offline reproducibility of model training and evaluation across Python/package versions

The real current suite also needs to be run on a controlled root to avoid unrelated workspace directories causing collection issues.

## 7. Conflicts between the paper narrative and implementation

The repository appears to mix a prototype app with a research narrative. The main conflicts are:

- synthetic data is treated as if it were clinical or real-world evidence in some outputs and descriptions
- model comparisons reference multiple algorithms without a single current runtime source of truth
- the app’s user-facing claims can exceed the evidence level of the training data and evaluation artifacts
- some documentation and model metadata are experiment outputs rather than product-grade runtime specifications

In short, the project is useful as a prototype but not yet scientific or clinical-grade evidence.

## 8. Repair plan (prioritized)

1. Establish a single canonical model and dataset provenance contract.
2. Stabilize package imports and pytest collection to a single project root.
3. Fix ORM compatibility issues and route/session handling that cause runtime crashes.
4. Re-run the relevant tests as a controlled regression suite.
5. Harden document and AI safety boundaries.
6. Reconcile documentation and paper claims with the real synthetic-data and prototype status.
7. Only after that, consider production deployment or publication-grade claims.

## 9. Current status

### Evidence update (2026-10-17)

The route tests now use the actual authentication contract: registration requires consent and stores a Werkzeug password hash; login accepts the email shown by the UI; the patient predictor requires a valid logged-in user. The old `/predict` URL is only a protected redirect to `/ivf_predictor`; actual inference and complete numeric validation are performed by authenticated `POST /predict_ivf`.

Focused validation evidence:

- `tests/test_routes.py::TestRoutes::test_login_success`: passed using a real `User` record and its password hash.
- `tests/test_integration.py::TestIntegration::test_complete_user_registration_login_flow`: passed with the registration consent field and email login.
- Five targeted predictor, login, wellness, and compatibility-route tests passed together.
- The route/integration suite reported 26 passed before additional chatbot coverage was added. A subsequent run collected 27 tests but the terminal did not return a complete report, so the updated combined suite is not declared passing here.
- Chat fallback/input-validation and wellness-persistence tests passed in a focused run. The document owner-isolation test passed when rerun individually after pytest's Windows `tmp_path` setup failed in a grouped run.

### Model artifact evidence

- `predict.py` loads `models/ivf_success_model.pkl` and its adjacent metadata. The deployed file loads as `RandomForestClassifier`; its SHA-256 at audit time was `4b1fed2fe7d25e79e3716e03433c66b8b26fe49b6873788267c765b9535f9c16`.
- Runtime scikit-learn is 1.8.0, while unpickling warns the RandomForest/DecisionTree artifact was created with scikit-learn 1.7.2. Other startup artifacts produce 1.9.0-versus-1.8.0 warnings.
- The model metadata says 5,000 synthetic records and accuracy 0.737; root and `models/` metadata disagree on training date. `train_model.py` is not a reproduction script for this artifact: it falls back to a 300-row synthetic generator with a heuristic label and trains a 150-tree RandomForest, while the deployed metadata says 200 trees.
- The checked-in comparison report records synthetic external-validation metrics, including Logistic Regression ROC-AUC 0.6850 and accuracy 0.5180; Random Forest external accuracy 0.5100 and a 0.4884 train/external accuracy gap. These are existing report values, not newly reproduced results, and the report itself warns the validation set is synthetic and not clinical evidence.
- Consequently, neither 0.737 metadata accuracy nor comparison report metrics are currently reproducible from a single documented canonical training/evaluation pipeline. No clinical validity is established.

### Security and workflow updates

- Removed the hard-coded Gemini API key from both copies of the unused standalone `ivf_chatbot.html`; their requests now target the authenticated same-origin chat/TTS APIs. Because the credential was committed in source, removal does not revoke it: rotate/revoke it with the provider.
- Chat API and TTS now reject non-object, blank, non-string, and over-4,000-character input. Chat responses receive a research/demo medical disclaimer. Focused fallback and persistence coverage passed.
- Added document-download ownership regression coverage; it passed individually. The upload/OCR workflow has focused tests for extraction, duplicate detection, old-report handling, unit/date validation, history, and prediction wiring. Full OCR suite status remains pending.
- Cycle and medication persistence routes exist, but automated persistence and cross-user isolation coverage remains incomplete. Authentication/session coverage is also not comprehensive.
- The project has no root `package.json`; there is no frontend package build to run. Templates and static JavaScript are served directly by Flask.

### Remaining work; project is not complete

1. Capture a definitive full `pytest tests` exit status and repair any real failures without excluding tests.
2. Run the full document workflow suite and add/verify cross-user document access cases for analyze/share endpoints as applicable.
3. Establish one canonical training/evaluation script, dataset provenance, pinned training environment, and reproducible metrics; reconcile both metadata copies. Do not present synthetic metrics as clinical evidence.
4. Add and run persistence/authorization tests for IVF cycles and medication reminders, plus user-data isolation across dashboards and APIs.
5. Review all secret-bearing source/history, rotate the exposed provider credential, and verify production secret/session configuration. The development fallback secret key remains unsafe for deployment.
6. Continue chatbot safety and privacy review, including provider data handling and user-facing medical guidance.
7. Reconcile paper/research claims to the demonstrated implementation and synthetic-data limits.

This remains a research/demo prototype. No clinical readiness or scientifically validated prediction performance is claimed.
