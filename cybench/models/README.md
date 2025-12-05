# Models

1. Model API: The `BaseModel` class implements the API. All models are required to implement `fit` and `predict_batch` methods. For models that work with a Dataset object, they can override the `predict` method.
2. Existing models
    a. Naive models: A naive model that predicts the training set average is included.

    b. Trend models: A trend model that supports linear and quadratic trend has been implemented. Trend estimation part is kept separate to add a different trend function without having to create a new model class.

    c. Ridge model: Ridge model from scikit-learn acts as a linear baseline with expert-designed features.

    d. Random Forest model. Ramdom Forest from scikit-learn acts as a non-linear baseline with expert-designed features. Both `Ridge` and `RandomForestRegressor` can be passed to `SklearnModel` as the sklearn estimator without having to change other code. Hyperparameter optimization is possible by passing the search space to `fit` method and setting the `optimize_hyperparameters` flag to True.

    e. XGBoost model. XGBoost (gradient boosting trees) provides a stronger non-linear baseline using the same expert-designed features and BaseSklearnModel pipeline, with hyperparameters tuned via grouped cross-validation.

    f. LSTM. An example LSTM model is included as a deep learning (RNN) or representation learning baseline.

    g. InceptionTime. An example InceptionTime model is included as a deep learning (1D-CNN) or representation learning baseline.

    h. Transformer. An example Transformer model is included as a deep learning (Transformer) or representation learning baseline.

    i. Residual models: All the models c-g have their residual modeling counterparts. Residual models subtract the prediction of a Linear Trend model from b and learn to predict or forecast the residual (i.e. yield - trend). The idea is to remove the location and time dependent component of yields. The seasonal predictors included in CY-Bench are likely to better forecast the year-to-year variability in yields  than yields themselves.
