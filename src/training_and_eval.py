model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=750,     
    learning_rate=0.03436512795332618,
    max_depth=4,
    colsample_bytree = 0.7551147066267492,
    subsample = 0.6978720555365889,
    reg_alpha = 2.2896781459596376e-05,
    reg_lambda = 2.0598804894156064,
    random_state=42
)

model.fit(X_train, Y_train,eval_set = [(X_test,Y_test)],verbose = False)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

Y_true_test = Y_scaler.inverse_transform(Y_test.reshape(-1, 1)).ravel()
Y_pred_test = Y_scaler.inverse_transform(test_pred.reshape(-1, 1)).ravel()

mae_test = mean_absolute_error(Y_true_test/1000,Y_pred_test/1000)
mse_test = mean_squared_error(Y_true_test/1000, Y_pred_test/1000)
print(Y_scaler.scale_[0])
print(f"{mae_test} MeV")
print(f"{mse_test} MeV2")
