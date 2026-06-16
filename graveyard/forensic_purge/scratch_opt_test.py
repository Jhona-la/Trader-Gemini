import optuna

def objective(trial):
    raise optuna.exceptions.TrialPruned()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2)

try:
    print("best_trials boolean:", bool(study.best_trials))
except Exception as e:
    print("Exception on best_trials:", e)

try:
    print("best_trial:", study.best_trial)
except Exception as e:
    print("Exception on best_trial:", e)
