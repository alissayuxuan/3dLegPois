import optuna
import json
import os
import subprocess
import copy
from pathlib import Path

BASE_CONFIG_PATH = "hyperparam_tuning/configs/femur_base_config.json"
TEMP_CONFIG_DIR = "hyperparam_tuning/temp_configs"
os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

METRIC_FILE_NAME = "best_metric.json"

def set_nested(config, key_path, value):
    """Setzt einen verschachtelten Wert in einem Dict über einen Pfad."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        config = config[key]
    config[keys[-1]] = value

def run_experiment_and_get_metric(config_path):
    """Trainiere Modell über train.py und hole Val Loss aus der Log-Datei."""
    with open(config_path) as f:
        config = json.load(f)

    subprocess.run(["python", "train.py", "--config", config_path])
    
    log_dir = os.path.join(config["path"], config["name"])

    metric_file = os.path.join(log_dir, METRIC_FILE_NAME)

    if os.path.exists(metric_file):
        with open(metric_file) as f:
            return json.load(f).get("val_loss", float("inf"))
        
    return float("inf")  # falls kein Ergebnis vorliegt

def objective(trial):
    # Lade Basisconfig
    with open(BASE_CONFIG_PATH) as f:
        config = json.load(f)

    # === Parameter Search Space definieren ===

    lr_coarse = trial.suggest_float("coarse_lr", 1e-5, 1e-3, log=True)
    lr_refine = trial.suggest_float("refine_lr", 1e-6, 1e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    weight_coarse = trial.suggest_float("loss_weight_coarse", 0.5, 2.0)
    weight_refine = trial.suggest_float("loss_weight_refine", 0.5, 2.0)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    #num_heads = trial.suggest_int("num_heads", 2, 8)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    mlp_dim = trial.suggest_categorical("mlp_dim", [256, 512, 1024])
    total_iters = trial.suggest_int("total_iters", 10, 50)

    # === Config anpassen ===

    set_nested(config, "module_config.params.coarse_config.params.lr", lr_coarse)
    set_nested(config, "module_config.params.refinement_config.params.lr", lr_refine)
    set_nested(config, "module_config.params.refinement_config.params.dropout", dropout)
    set_nested(config, "module_config.params.loss_weights", [weight_coarse, weight_refine])
    set_nested(config, "module_config.params.refinement_config.params.num_layers", num_layers)
    set_nested(config, "module_config.params.refinement_config.params.num_heads", num_heads)
    set_nested(config, "module_config.params.refinement_config.params.mlp_dim", mlp_dim)
    set_nested(config, "module_config.params.scheduler_config.params.total_iters", total_iters)

    # === Logging Pfad setzen ===
    config_name = f"trial_{trial.number}"
    config["path"] = os.path.join("hyperparam_tuning/optuna_runs", config_name)
    config["name"] = config_name

    # === Temporäre Config schreiben ===
    temp_config_path = os.path.join(TEMP_CONFIG_DIR, f"{config_name}.json")
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=4)

    # === Training starten und Ergebnis evaluieren ===
    val_loss = run_experiment_and_get_metric(temp_config_path)
    return val_loss  # Minimierung

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40)

    print("Beste Konfiguration:")
    print(study.best_trial.params)

    # Beste Config separat speichern
    with open("best_config_optuna.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
