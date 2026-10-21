# Test and Verification Report

Date: 2026-10-17
Environment: Windows, project `.venv`, Python 3.13.2, pytest 9.1.1, scikit-learn 1.8.0.

## Route failure diagnosis and corrections

Initial reproduced route-suite summary: 1 failed, 25 passed, 20 warnings. The only failing test was `tests/test_routes.py::TestRoutes::test_login_success`. Its mocked `routes.User` was not the symbol used by the actual Flask handler in `main.py`; the handler issued a real query against a missing `user` table. The failure was therefore a brittle test mock. A subsequent isolated run also established that the old assertion read Flask's `session` proxy outside a request context. The test now creates a real `User`, verifies the Werkzeug password hash, logs in using the email field used by the UI, and inspects the session inside a client context.

Other outdated test assumptions were corrected against the implementation:

- Registration requires `agree_terms`; tests now submit the consent field and verify a real password hash.
- Login form and product UI use email; username-only test login was removed. Username-based login is not implemented as product behavior.
- `/ivf_predictor` is a protected patient page. `/predict` remains a protected legacy redirect; the actual validated prediction API is `POST /predict_ivf`.
- The wellness page’s visible heading is “Wellness & Mood Tracking”; tests assert that visible text rather than an injected hidden heading.
- The compatibility `routes.py` module imports only model classes and no longer imports `predict.py` (which imports `main.py` and risks circular app initialization).

## Test runs

| Command / test | Result |
|---|---|
| `python -m pytest tests/test_routes.py::TestRoutes::test_login_success -vv --tb=long` before test correction | Failed: test accessed `session` outside request context; patched module symbol did not match `main.User` lookup |
| `python -m pytest tests/test_routes.py::TestRoutes::test_login_success -vv --tb=long` after correction | 1 passed, 7 warnings |
| `python -m pytest tests/test_integration.py::TestIntegration::test_complete_user_registration_login_flow -vv --tb=long` | 1 passed, 7 warnings |
| Targeted predictor/login/wellness/route compatibility tests | 5 passed, 13 warnings |
| Earlier corrected `tests/test_integration.py tests/test_routes.py` run | 26 passed, 17 warnings; this was before the new chatbot regression test was added |
| Focused chatbot fallback/input-validation and wellness-persistence tests | 2 passed |
| `tests/test_document_workflow.py::test_document_download_is_restricted_to_owner` | 1 passed, 9 warnings |
| Latest exact verbose route-suite run | Collected 27 tests, but terminal output stopped before the final result; not treated as verified pass |
| Latest complete `pytest tests` run | Started without a complete exit summary available in the tool output; not treated as verified pass |

No test was excluded by `pytest.ini` within the intended `tests/` directory. Full-suite completion remains a blocker; do not infer a green result from partial output.

## Model evidence (inspection, not a new benchmark)

- Loaded runtime artifact: `models/ivf_success_model.pkl`, `RandomForestClassifier`.
- SHA-256: `4b1fed2fe7d25e79e3716e03433c66b8b26fe49b6873788267c765b9535f9c16`.
- Runtime scikit-learn: 1.8.0. Pickle emits an incompatibility warning for artifact estimators built with 1.7.2; other loaded model files warn about 1.9.0.
- Paired metadata states 5,000 synthetic training records, 200 trees, accuracy 0.737, and a 2026-02-18 timestamp. Root `ivf_model_metadata.json` states a different date, 2024-02-16.
- `train_model.py` currently generates 300 heuristic-label synthetic records and trains 150 trees if database labels are unavailable. It cannot reproduce the deployed metadata/artifact as described.
- Existing `final_ivf_model_report.txt` reports synthetic external validation: Logistic Regression accuracy 0.5180 and ROC-AUC 0.6850; Random Forest accuracy 0.5100, with train accuracy 0.9984 and train/external gap 0.4884. These existing file values were not re-run in this session and are not clinical evidence.

## Warnings and known gaps

- scikit-learn pickle-version mismatch warnings (1.7.2/1.9.0 artifacts loaded under 1.8.0).
- SQLAlchemy `Query.get()` legacy deprecation warnings.
- Startup may warn that optional XGBoost is unavailable; the prediction service falls back to a rule-based estimate. The existence of this fallback does not validate it scientifically.
- Pytest's Windows `tmp_path` fixture failed during one combined focused run before test execution. The ownership test was made self-cleaning under `app.instance_path` and passed independently.
- The bundled project has no root `package.json`; no npm frontend build applies.
- OCR full-suite completion, broader persistence/isolation coverage, complete security review, reproducible model pipeline, paper alignment, and clinical validation remain unfinished.

## Changed files in this continuation

- `main.py`: restore email-only authentication and protected predictor access; validate chatbot/TTS payloads; append chatbot research/demo disclaimer.
- `routes.py`: remove circular-import-prone `predict` and unused password helper compatibility imports.
- `ivf_chatbot.html`, `templates/ivf_chatbot.html`: remove hardcoded provider credential and route requests through authenticated same-origin APIs.
- `tests/test_integration.py`, `tests/test_routes.py`: align tests with current registration/login/predictor behavior and verify persistence/visible content.
- `tests/test_document_workflow.py`: add document download owner-isolation coverage with test-local cleanup.
- `PROJECT_AUDIT.md`: update evidence and outstanding work.

## Status

Not complete. Do not deploy as a clinical system or present the model as scientifically validated.