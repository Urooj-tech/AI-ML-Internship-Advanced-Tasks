# Telco Customer Churn Prediction

##  Objective
The objective of this project is to build an **end-to-end machine learning pipeline** for predicting customer churn using the Telco Customer Churn dataset. The pipeline is designed to be reusable and production-ready.

##  Dataset
- **Source:** Telco Churn dataset (Kaggle)
- **Target Variable:** `Churn` (Yes/No)
- **Features:** Customer demographics, account details, services subscribed, tenure, monthly charges, etc.

##  Methodology / Approach
1. **Data Loading & Preprocessing**
   - Handle missing values
   - Encode categorical variables using OneHotEncoder
   - Scale numerical features using StandardScaler

2. **Model Development**
   - Logistic Regression
   - Random Forest Classifier

3. **Hyperparameter Tuning**
   - Used `GridSearchCV` for hyperparameter optimization

4. **Evaluation**
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-score)
   - Confusion Matrix

5. **Export**
   - The best pipeline is saved as `churn_pipeline.joblib` for future use.

##  Key Results / Observations
- Random Forest generally achieved better performance than Logistic Regression.
- Feature preprocessing (scaling + encoding) significantly improved model accuracy.
- Final trained pipeline can be directly loaded and used for prediction on new customer data.

##  Repository Structure
```
├── telco_churn_pipeline.ipynb   # Jupyter Notebook with full workflow
├── telco_churn_pipeline.py      # Python script version
├── telco_churn.csv              # Dataset (not included in repo, add manually from kaggle)
├── churn_pipeline.joblib        # Exported trained model
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🚀 How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/telco-churn-pipeline.git
   cd telco-churn-pipeline
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run Jupyter Notebook  
   ```bash
   jupyter notebook telco_churn_pipeline.ipynb
   ```

or run the script directly:  
   ```bash
   python telco_churn_pipeline.py
   ```

## ✅ Final Insights
- Churn prediction helps businesses identify at-risk customers early.  
- A reusable ML pipeline ensures consistent preprocessing and model application in production.  
- Further improvement could be achieved by testing advanced models (e.g., Gradient Boosting, XGBoost).  

---
🔹 Author: *Urooj Fatima*  
