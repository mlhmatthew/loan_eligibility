# Loan Eligibility Prediction App

## Overview

This is a machine learning-based web application that predicts whether a loan applicant is likely to be approved for a loan based on their personal and financial information. Built with Streamlit and powered by a logistic regression model, the application demonstrates a complete machine learning pipeline — from data preprocessing to real-time prediction.

Users can interact with the app by inputting details such as income, loan amount, credit history, and employment status to receive a prediction about their loan approval eligibility.

## Live Demo

Try the app live: [https://loaneligibility-vskjfkwpg7ysnw2uqhuymp.streamlit.app/](https://loaneligibility-vskjfkwpg7ysnw2uqhuymp.streamlit.app/)

## Features

- Trained logistic regression model using real-world-like loan data
- Clean data preprocessing with handling of missing values and categorical encoding
- Real-time prediction based on user inputs
- Web interface built using Streamlit
- Model evaluation with accuracy and confusion matrix display

## Dataset

The dataset used for training the model is `credit.csv`, which contains personal and financial information about past loan applicants. Features include:

- `ApplicantIncome`, `CoapplicantIncome`
- `LoanAmount`, `Loan_Amount_Term`
- `Credit_History`, `Gender`, `Married`
- `Education`, `Self_Employed`, `Property_Area`
- Target: `Loan_Approved` (1 = Approved, 0 = Not Approved)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mlhmatthew/loan-eligibility-app.git
   cd loan-eligibility-app
   ```

2. Create a virtual environment:
   ```bash
   conda create -n mlenv python=3.9
   conda activate mlenv
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Launch the Streamlit app:
   ```bash
   streamlit run app/loan_eligibility_app.py
   ```

2. Open your browser and go to:
   ```
   http://localhost:8501
   ```

3. Enter the required applicant details and click **Predict** to get the loan approval result.

## File Structure

```
CST2213_LOAN_ELIGIBILITY/
│
├── app/
│   └── loan_eligibility_app.py          # Streamlit web application entry point
│
├── loan_eligibility/                    # Core ML logic
│   ├── __init__.py                      # Makes the directory a package
│   ├── data_loader.py                   # Loads the dataset
│   ├── model.py                         # Trains and evaluates the model
│   ├── predict.py                       # Prediction logic for new applicants
│   └── preprocess.py                    # Data cleaning and encoding
│
├── credit.csv                           # Loan dataset
├── README.md                            # Project documentation
├── requirements.txt                     # List of Python dependencies
└── .gitignore                           # Git ignore file

```

## Requirements

- Python 3.9+
- scikit-learn
- pandas
- numpy
- streamlit

Install all dependencies using:
```bash
pip install -r requirements.txt
```
