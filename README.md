# Conformal Prediction - Uncertainty Quantification

This project explores and implements Conformal Prediction methods to quantify uncertainty in Machine Learning models. The goal is to provide reliable predictions accompanied by clear statistical guarantees, applicable to both continuous/discrete data and categorical data.

## Use Cases and Methods

The project is divided into two main areas of study:

1. Regression: Hospital Length of Stay Estimation
The objective is to provide guaranteed prediction intervals, expressed in full days, to facilitate bed capacity management.
- Conformal Quantile Regression (CQR): Baseline approach using quantile regression.
- Normalized Conformal Prediction (SCP): A two-stage (Locally Scaled) modeling approach to adapt the interval width to the clinical complexity of each patient and handle the discrete nature of count data.

2. Ordinal Classification: # TODO

## Project Structure

* `data/`: Contains raw/processed datasets, serialized models (.joblib), and output plots.
* `src/`: Reusable core modules.
  * `preprocessing.py`: Data cleaning and transformation.
  * `models.py`: Machine Learning algorithm definition and training.
  * `conformal_regression.py`: Implementation of calibration algorithms for regression.
  * `evaluation.py`: Calculation of coverage metrics and fairness auditing.
  * `visualization.py`: Generation of analytical plots.
* `notebooks/`: Experiments and initial data exploration.
* `main_regression.py`: Entry point for executing the regression pipeline.
* `main_classification.py`: Entry point for executing the classification pipeline.

## Installation

Create a virtual environment, we recommend using `uv`:
```bash
uv venv
source .venv/bin/activate
```

Install the required dependencies using the provided requirements file:
```bash
uv pip install -r requirements.txt
```

## Usage

The scripts use `argparse` to configure command-line executions.

To run the regression pipeline (normalized method with group fairness audit):
```bash
python main_regression.py --method normalized --grouped
```

To run the baseline quantile regression approach:

```bash
python main_regression.py --method cqr
```

To run the classification pipeline:
```bash
# TODO
```