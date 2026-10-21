import json
import traceback
import sys
import runpy
from pathlib import Path

# Load the training module by file path to avoid package import issues
ml_models_dir = Path(__file__).parent / "ChoreCompanion" / "ml_models"
module_path = ml_models_dir / "model_training.py"
# Ensure local ml_models directory is on sys.path so its local imports resolve
sys.path.insert(0, str(ml_models_dir))
if not module_path.exists():
    print(f"Training module not found: {module_path}")
    sys.exit(1)

try:
    module_globals = runpy.run_path(str(module_path))
    if 'train_all_models' not in module_globals:
        print('train_all_models not found in training module')
        sys.exit(1)
    results = module_globals['train_all_models']()
    print(json.dumps(results, default=str, indent=2))
except Exception:
    traceback.print_exc()
    sys.exit(1)
