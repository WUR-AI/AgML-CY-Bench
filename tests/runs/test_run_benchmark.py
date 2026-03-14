# from cybench.runs.run_benchmark import run_benchmark
#
#
# def test_run_benchmark():
#     # skipping some models
#     baseline_models = [
#         "AverageYieldModel",
#         "LinearTrend",
#         "SklearnRidge",
#         "RidgeRes",
#         "LSTM",
#     ]
#     nn_models_epochs = 5
#     run_benchmark(
#         run_name="maize_NL",
#         dataset_name="maize_NL",
#         baseline_models=baseline_models,
#         sel_years=[2010, 2011],
#         nn_models_epochs=nn_models_epochs,
#     )
