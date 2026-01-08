# Customer Churn Prediction 📉

Predicting customer churn using machine learning to help businesses retain customers and reduce revenue loss.

---

## 🚀 Project Overview

Customer churn is a critical problem for subscription-based and service-driven businesses.  
This project builds a **complete machine learning pipeline** to predict whether a customer is likely to churn based on historical behavioral and demographic data.

The focus is on:
- Data preprocessing
- Feature engineering
- Model training & evaluation
- Real-world business interpretability

---

## 🧠 Problem Statement

Businesses lose a significant amount of revenue due to customer churn.  
Identifying **high-risk customers in advance** allows companies to:
- Take preventive actions
- Improve customer retention
- Optimize marketing costs

---

## 🗂️ Project Structure


customer-churn-prediction/
│
├── data/
│ └── churn.csv
│
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── evaluation.py
│ └── main.py
│
├── notebooks/
│ └── exploration.ipynb
│
├── requirements.txt
└── README.md



---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **ML Models:** Logistic Regression, Random Forest  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  

---

## 🔍 Key Features

- ✔️ Cleaned and preprocessed raw customer data  
- ✔️ Handled missing values & categorical encoding  
- ✔️ Feature scaling and selection  
- ✔️ Trained multiple ML models  
- ✔️ Compared performance using proper metrics  
- ✔️ Business-oriented churn interpretation  

---

## 📊 Model Evaluation

The trained models were evaluated using:
- Confusion Matrix
- Precision–Recall tradeoff
- F1-score to handle class imbalance

> The final model balances **recall (catching churners)** and **precision (avoiding false alarms)**.

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/Jugalbrahmkshatriya/customer-chrun-prediction
cd customer-churn-prediction
pip install -r requirements.txt
python src/main.py


## 🤝 Contributions

Feel free to open issues or submit pull requests.
If you liked this project, don’t forget to ⭐ the repo.
