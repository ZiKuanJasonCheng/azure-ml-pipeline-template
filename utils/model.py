import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def xgboost_model(x_train, y_train, x_val, y_val) -> (xgb.XGBClassifier, int):

    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.05, 0.01, 0.001],
        'n_estimators': [50, 100, 200],
    }

    # Specify StratifiedKFold as the cross-validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create XGBoost model 1
    clf = xgb.XGBClassifier(enable_categorical=True)
    grid_search_model = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search_model.fit(x_train, y_train)

    # Get the best parameters
    best_params = grid_search_model.best_params_
    print(best_params)

    best_model = xgb.XGBClassifier(**best_params, enable_categorical=True)
    best_model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(x_val)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Best Model Accuracy: {accuracy}")

    return best_model, accuracy