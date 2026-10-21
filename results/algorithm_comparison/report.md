# Algorithm Comparison / Ablation Study

> Research prototype only. The experiment uses synthetic project data; results are not clinical evidence.

## Objective
Compare common classifiers under one reproducible dataset, split, feature set, and training-only preprocessing setup.

## Experimental setup
- Dataset: `ivf_5000_dataset.csv` (5000 rows, 9 columns).
- Features: age, bmi, amh, fsh, previous_ivf, stress, sleep_hours, exercise_min; target: `IVF_Success` (binary classification).
- Target counts: {'0': 1649, '1': 3351}; missing values: 0; duplicate rows: 0.
- Preprocessing: median imputation followed by standard scaling in every model pipeline. Each fold fits preprocessing on its training partition only.
- Hold-out: stratified 80/20 split, random seed 42; five-fold shuffled stratified CV runs on training data only.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, CV mean/std, and train/test accuracy gap.
- Leakage review: target excluded and no target-derived input column identified. However, `generate_ivf_5000.py` creates labels from a hand-written score over the input features and samples them using score-based probabilities. The target is therefore synthetic and structurally related to these features; this is a major validity limitation, though not direct target-column leakage.

## Algorithms compared
Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and Gradient Boosting.

## Results
| Algorithm           | Accuracy | Precision | Recall | F1-score | ROC-AUC | Train Accuracy | Train F1-score | Train-Test Accuracy Gap | CV Accuracy Mean | CV Accuracy Std | CV F1 Mean | CV F1 Std | CV Train Accuracy Mean | Confusion Matrix TN,FP,FN,TP |
| ------------------- | -------- | --------- | ------ | -------- | ------- | -------------- | -------------- | ----------------------- | ---------------- | --------------- | ---------- | --------- | ---------------------- | ---------------------------- |
| Decision Tree       | 0.6410   | 0.7338    | 0.7284 | 0.7311   | 0.5960  | 1.0000         | 1.0000         | 0.3590                  | 0.6278           | 0.0189          | 0.7213     | 0.0149    | 1.0000                 | 153,177,182,488              |
| Gradient Boosting   | 0.7310   | 0.7548    | 0.8866 | 0.8154   | 0.7335  | 0.7847         | 0.8522         | 0.0537                  | 0.7328           | 0.0041          | 0.8168     | 0.0028    | 0.7970                 | 137,193,76,594               |
| KNN                 | 0.6910   | 0.7397    | 0.8313 | 0.7829   | 0.6712  | 0.7983         | 0.8587         | 0.1073                  | 0.7140           | 0.0089          | 0.8001     | 0.0064    | 0.7964                 | 134,196,113,557              |
| Logistic Regression | 0.7260   | 0.7538    | 0.8776 | 0.8110   | 0.7344  | 0.7428         | 0.8233         | 0.0168                  | 0.7390           | 0.0094          | 0.8206     | 0.0064    | 0.7429                 | 138,192,82,588               |
| Random Forest       | 0.7390   | 0.7579    | 0.8970 | 0.8216   | 0.7244  | 0.9545         | 0.9671         | 0.2155                  | 0.7350           | 0.0041          | 0.8194     | 0.0026    | 0.9604                 | 138,192,69,601               |
| SVM                 | 0.7400   | 0.7482    | 0.9224 | 0.8262   | 0.7194  | 0.7630         | 0.8413         | 0.0230                  | 0.7363           | 0.0086          | 0.8239     | 0.0049    | 0.7646                 | 122,208,52,618               |

## Findings
- Highest Accuracy: SVM (0.7400).
- Highest Precision: Random Forest (0.7579).
- Highest Recall: SVM (0.9224).
- Highest F1-score: SVM (0.8262).
- Highest ROC-AUC: Logistic Regression (0.7344).
- Possible overfit gap (descriptive flag at accuracy gap ≥ 0.10): Decision Tree (0.359), KNN (0.107), Random Forest (0.216)
- False negatives are shown in each confusion matrix (lower-left cell). Their consequences require clinical context and are not assessed by this experiment.
- CV standard deviations indicate variation across the five folds; they do not replace external validation.

## Existing application model
Existing application model: RandomForestClassifier according to models/ivf_model_metadata.json; main.py loads models/ivf_success_model.pkl. Its test metrics are descriptive only because its original training records are not documented; overlap with this dataset may make them optimistic.

Reference metrics (interpret with the training-overlap caveat above):
| Algorithm                        | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| -------------------------------- | -------- | --------- | ------ | -------- | ------- |
| Existing saved application model | 0.7370   | 0.7560    | 0.8970 | 0.8205   | 0.7245  |

## Conclusion
These measurements compare experimental predictive performance on the available synthetic dataset. They do not establish clinical validity or support clinical use; external validation on appropriately governed real-world data would be required.
