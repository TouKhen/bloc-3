# Customer Churn Analysis Project

This project analyzes customer churn data using machine learning techniques to
identify patterns and predict customer churn behavior.

## Project Structure

The project is organized into several key directories:

### `/data`

- Contains the raw dataset `telco-customer-churn.csv` used for analysis

### `/docs`

- Generated documentation files in HTML format
- Includes separate documentation for data processing, feature engineering,
  models, and visualizations
- Contains `search.js` for documentation search functionality

### `/notebooks`

- Jupyter notebooks for exploratory data analysis:
    - `data-explo.ipynb`: Initial data exploration
    - `data-explo-clean.ipynb`: Cleaned version with refined analysis

### `/reports`

- Contains the `/figures` subdirectory for storing generated plots and
  visualizations

### `/src`

Core Python modules implementing the main functionality:

- `data.py`: Data loading and basic cleaning operations
- `features.py`: Feature engineering and transformation
- `models.py`: Machine learning model implementations
- `viz.py`: Visualization functions

### Root Directory

- `Bloc 3 - Brief Introduction au Machine Learning.pdf`: Project documentation
- `README.md`: Project overview and setup instructions


## Project install
```bash
git clone https://github.com/TouKhen/bloc-3.git
python -m venv venv
cd venv
```

Dependency install :
```bash
pip install -r requirements.txt
```