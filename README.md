# Week 6: Model Deployment & Serialization

I built this project as the final capstone for my Applied ML Internship. It takes a house-price regression model from training in a notebook to making predictions through a Streamlit app.

The notebook trains a Random Forest regressor, saves the trained model together with its scaler and feature names as a single `joblib` artifact, and reloads it to check that it can make a prediction. The Streamlit app loads that artifact and estimates a property's market value from its area, number of bedrooms and bathrooms, and age.

## Project files

- `Week6_Model_Deployment.ipynb` — trains and evaluates the model, saves the artifact, and checks a sample prediction.
- `app.py` — Streamlit interface for entering property details and viewing an estimated price.
- `house_prices.csv` — local dataset used by the notebook; no Kaggle API credentials are needed.
- `house_price_pipeline.joblib` — serialized model, scaler, and feature names used by the app.

## Run the app

Install the required packages:

```bash
pip install pandas scikit-learn joblib streamlit
```

Then launch the app from the project folder:

```bash
streamlit run app.py
```

The included `house_price_pipeline.joblib` artifact lets the app run without retraining. To regenerate it, open and run `Week6_Model_Deployment.ipynb` with the dataset in the same folder.

## Note

The displayed price is a model estimate based on historical sales data, not a formal property appraisal.
