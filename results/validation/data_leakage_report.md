# Data Leakage Review

- **Target excluded from comparison inputs — PASS:** Declared feature list: ['age', 'bmi', 'amh', 'fsh', 'previous_ivf', 'stress', 'sleep_hours', 'exercise_min']; target: IVF_Success.
- **Target-derived columns — PASS:** No additional target-derived column appears in this CSV schema.
- **Duplicate records — PASS:** 0 exact duplicate rows beyond first copies; 0 repeated feature rows beyond first copies.
- **Preprocessing/split order — PASS:** Comparison code uses a stratified hold-out and Pipeline/cross-validation, fitting preprocessing within each training fold.
- **Future/post-outcome columns — PASS:** CSV contains only baseline/lifestyle-like synthetic inputs and IVF_Success; no date or future workflow field is present.
- **Algorithm-comparison data linkage — INFO:** The existing comparison remains unchanged and loads ivf_5000_dataset.csv directly. This validation creates a separate validated copy; it does not alter the comparison's input path. The validated file retains all source rows and feature/target values.
- **Synthetic target dependence — WARNING:** Dataset generator computes success_score from hand-written thresholds over the same input features, then samples success_prob from that score. This is structural target dependence, not direct target-column leakage; measurements reflect the synthetic generating pattern, not observed clinical outcomes.
