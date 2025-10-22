import optuna
import xgboost as xgb
def objective(trial):
    param = {
         "device" : "cuda",
         "n_estimators" : trial.suggest_int("n_estimators", 200,1000,step = 50),
         "learning_rate" : trial.suggest_float("learning_rate", 1e-4,5e-2,log = True),
         "max_depth" : trial.suggest_int("max_depth", 2,10),
         "colsample_bytree" : trial.suggest_float("colsample_bytree",0.5,1.0),
         "subsample" : trial.suggest_float("subsample",0.5,1.0),
         "reg_alpha" : trial.suggest_float("reg_alpha",1e-8, 10.0, log=True),
         "reg_lambda" : trial.suggest_float("reg_lambda",1e-8, 10.0, log=True)
     }
    
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, Y_train,eval_set = [(X_test,Y_test)],verbose = False)
    preds = model.predict(X_test)
    mse_scaled = mean_squared_error(Y_test,preds)
    mse_original = mse_scaled * ((Y_scaler.scale_[0]/1000)**2)
    return mse_original

study = optuna.create_study(direction = "minimize")
n_trials = input("Enter number of trials used for study : ")
study.optimize(objective, n_trials = n_trials)
print(study.best_value)
print(study.best_params)
