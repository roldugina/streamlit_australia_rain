# Building a Web Application with Streamlit for Predicting Rain in Australia

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://australia-rain.streamlit.app/)

This project focuses on building a web application that predicts whether **rainfall will occur** in a given region of Australia, with the goal of demonstrating the complete machine learning pipeline, from **data preprocessing** to **prediction based on the entered data**, and **deploying the trained model** using the Streamlit library.

## Dataset:

The Kaggle dataset comprises about **10 years** of daily weather observations from numerous locations across Australia. 
You can download it here: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

## Project Overview

The task is a **binary classification** problem, with **imbalanced classes** (one class is **3.46** times more frequent than the other). The dataset is heterogeneous, consisting of 16 numerical and 5 categorical features.

For the metrics in the rain prediction task, I have chosen **F1-Score** and **recall** to prioritize minimizing false negatives. I consider it more critical to avoid missing instances of rain, as this could lead to significant costs.

## Technologies Used:

   - **Python** for data processing and machine learning.
   - **Streamlit** for web app deployment.
   - **Scikit-learn** for building and evaluating machine learning models.
   - **Pandas** for data manipulation.
   - **XGBoost** and **Hyperopt** for xgboost model training and hyperparameter optimization.
   - **imblearn** for handling imbalanced dataset.
   - **Joblib** for saving a pre-trained model.
   
  Output: The model predicts rainfall occurrence and provides the rain probability.
  
  The project uses a virtual environment for dependency management.

## Conclusions

The model `XGBoost` with `Hyperopt` optimization and threshold adjustment demonstrated notable improvements over `baseline` (`LogisticRegression`). `AUROC` increased from `0.872397` to `0.898868`, showing a better ability to distinguish between classes. `Recall` improved significantly from `0.460872` to `0.778088`, indicating a much better performance in identifying the minority class. Additionally, the `F1-score` rose from `0.569800` to `0.652678`, reflecting a more balanced performance between precision and recall. 

These results suggest that the use of `XGBoost` with `Hyperopt` optimization and threshold adjustment significantly enhanced the model's ability to correctly classify the minority class while maintaining a good overall performance.

![trained models](https://drive.google.com/file/d/1kQgYsYp6gFdJfpAsj4WPetK76qUK1aDQ/view?usp=sharing)

  [Try deployed app](https://australia-rain.streamlit.app/)
