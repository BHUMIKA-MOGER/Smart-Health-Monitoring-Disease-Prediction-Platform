# 🏥 Health Portal – AI-Based Disease Prediction System

An AI-powered healthcare web application built using **Flask** and **Machine Learning** that predicts diseases and assists in clinical decision-making.

---

## 🚀 Features

- 🫀 Heart Disease Prediction
- 🩸 Diabetes Detection
- 💊 Drug Recommendation & Dose Prediction
- 🧠 Symptom-Based Risk Analysis
- 📊 Clinical Risk Calculator
- 🔐 User Login & Dashboard

---

## 🧠 Machine Learning Models Used

- Random Forest
- Logistic Regression
- Multi-class Classification
- Label Encoding for categorical features

Pre-trained models are stored as `.pkl` files for fast inference.

---

## 🗂 Project Structure

```
HEALTH_PORTAL/
│<br>
├── app.py
├── drug_model.py
├── train_model.py
├── train_heart_model.py
│
├── templates/
│ ├── login.html
│ ├── dashboard.html
│ ├── diabetes.html
│ ├── drug_dose.html
│ ├── heart_disease.html
│ ├── heart_symptom_check.html
│ └── clinical_risk_calculator.html
│
├── static/
│ └── record.gif
│
├── models/
│ ├── *.pkl
│
├── datasets/
│ ├── *.csv
│
├── .gitignore
└── README.md
```


---

## 🛠 Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML / CSS

---

## ▶️ How to Run the Project

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/health-portal.git

---
2. Create virtual environment<br>
   ```python -m venv venv```

---
3. Activate environment<br>
   Windows:<br>
    ```
    venv\Scripts\activate
    ```
   ---
   Linux/Mac:
   ```
   source venv/bin/activate
    ```
---
4. Install dependencies<br>
   ```
   pip install flask scikit-learn pandas numpy
   ```

---
5. Run the application<br>
   ```
   python app.py
   ```

---
6. Open browser and visit:<br>
   ```
   http://127.0.0.1:5000/
   ```





