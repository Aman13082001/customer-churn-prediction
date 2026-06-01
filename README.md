# 📊 ChurnIQ — Customer Churn Intelligence Platform

> An end-to-end machine learning dashboard for predicting customer churn, comparing ML models, and driving data-driven retention strategies.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 🎯 Overview

ChurnIQ is a **production-style ML dashboard** built to monitor, analyze, and predict customer churn in a telecom company. It compares 4 machine learning models, provides actionable business-level churn insights, and simulates individual customer risk in real time.

**Built with:** Streamlit • scikit-learn • XGBoost • Plotly • Pandas • Joblib
**All free and open source — no paid APIs, no cloud costs.**

## 🚀 Features

- **5 Fully Interactive Pages**
  - 📊 Home — Executive dashboard with key metrics
  - 📁 Dataset Overview — Data quality, types, distributions, correlations
  - 📈 Churn Analysis — Patterns by contract, tenure, charges, services
  - 🤖 Model Performance — Compare 4 models, ROC curves, feature importance
  - 🔮 Customer Prediction — Risk simulator with retention recommendations

- **4 ML Models Compared**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost

- **Company-Grade UI**
  - Custom CSS styling (dark mode, brand colors)
  - Responsive charts with Plotly
  - Real-time model predictions with gauges
  - Branded sidebar and navigation
  - Professional data visualizations

- **Business Intelligence**
  - Churn rate trends by customer segment
  - Risk-based customer scoring
  - AI-driven retention recommendations
  - Model performance transparency (confusion matrix, ROC, feature importance)

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **81.2%** | **80.5%** | **78.3%** | **0.793** | **0.848** |
| Random Forest | 80.8% | 80.1% | 77.8% | 0.788 | 0.841 |
| Logistic Regression | 81.0% | 79.8% | 79.2% | 0.794 | 0.844 |
| Decision Tree | 79.3% | 78.2% | 76.5% | 0.773 | 0.821 |

## 💻 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/Aman13082001/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

### Docker Setup (Optional)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run PowerShell script on Windows
.\run_docker.ps1
```

## 📁 Project Structure

```
customer-churn-prediction/
├── README.md                      # This file
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Container configuration
├── docker-compose.yml             # Multi-container setup
├── run_docker.ps1                 # Windows Docker launcher
│
├── app/                           # Streamlit dashboard
│   ├── app.py                     # Main entrypoint + sidebar
│   ├── pages/                     # 5 dashboard pages
│   │   ├── page_01_home.py
│   │   ├── page_02_dataset_overview.py
│   │   ├── page_03_churn_analysis.py
│   │   ├── page_04_model_performance.py
│   │   └── page_05_prediction.py
│   └── utils/                     # Utilities & helpers
│       ├── data_utils.py          # Data loading & preprocessing
│       ├── model_utils.py         # Model loading & prediction
│       ├── ui_components.py       # Reusable UI components
│       └── plotting.py            # Chart helpers
│
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/
│       └── churn_cleaned.csv      # Cleaned dataset (7043 customers × 19 features)
│
├── models/                        # Trained ML artifacts
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── model_metadata.json
│
├── src/                           # Training pipeline
│   ├── models/
│   │   ├── train.py               # Model training (fixed CV leakage)
│   │   ├── evaluate.py            # Evaluation utilities
│   │   ├── predict.py             # Prediction helpers
│   │   └── config.py              # Training config
│   ├── features/
│   │   └── feature_engineering.py # Feature creation
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   └── visualization/
│       └── plots.py               # Plotting utilities
│
├── notebooks/                     # Jupyter explorations
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   └── 03_feature_engineering.ipynb
│
├── tests/                         # Unit tests
│   └── test_data_utils.py
│
├── assets/                        # Images & screenshots
│   └── images/
│
└── reports/                       # Generated reports
    ├── figures/
    └── model_results/
```

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Dashboard UI** | Streamlit | 1.35.0 |
| **ML Models** | scikit-learn, XGBoost | 1.4.2, 2.0.3 |
| **Data Processing** | pandas, numpy | 2.2.2, 1.26.4 |
| **Visualization** | Plotly | 5.22.0 |
| **Model Serialization** | joblib | 1.4.2 |
| **Containerization** | Docker | Latest |

## 📊 Key Insights

1. **Contract Type is #1 Lever**
   - Month-to-month customers churn at ~55% — 3.5× higher than 2-year contracts
   - Focus retention on contract upsells

2. **Early Engagement Critical**
   - Customers in first 12 months churn at 3× the tenured rate
   - Implement onboarding and activation campaigns

3. **Service & Pricing Factors**
   - Fiber optic customers show high churn (possible competition/pricing)
   - Higher monthly charges correlate with churn risk

4. **Model Performance**
   - XGBoost and Logistic Regression both achieve ~81% accuracy
   - ROC-AUC of 0.85 enables risk-based prioritization

## 🎓 What You'll Learn

- **Data Engineering** — Data loading, cleaning, validation, preprocessing
- **ML Workflows** — Train/test splits, cross-validation (without leakage), model evaluation
- **Model Selection** — Compare 4 algorithms, evaluate with multiple metrics
- **Web Dashboard** — Build interactive UIs with Streamlit
- **Software Engineering** — Modular code, reusable components, error handling
- **DevOps** — Docker containerization, reproducible deployments

## 🤖 Models Explained

### 1. Logistic Regression
- Fast, interpretable baseline
- Linear decision boundary
- Good for feature importance understanding

### 2. Decision Tree
- Simple, visual decision rules
- Prone to overfitting
- Useful for business rule extraction

### 3. Random Forest
- Ensemble of decision trees
- Reduces overfitting
- Feature importances via bootstrap aggregation

### 4. XGBoost
- Gradient boosting with regularization
- State-of-the-art performance
- Handles class imbalance well

## 🔄 Pipeline Highlights

✅ **Fixed Data Leakage**
- Cross-validation runs only on training data
- No information from test set used during CV

✅ **Standardized Imports**
- All imports use absolute paths: `from app.utils.data_utils import ...`
- Works everywhere (local, Docker, CI/CD)

✅ **Pinned Dependencies**
- All versions locked in `requirements.txt`
- Reproducible across environments

✅ **Error Handling**
- Graceful fallbacks if models not found
- Input validation in prediction
- Try/except blocks in all critical sections

## 📈 Usage Examples

### Explore the Dashboard
```
streamlit run app/app.py
→ Navigate to http://localhost:8501
→ Browse all 5 pages
```

### Train New Models
```bash
python src/models/train.py
# Saves artifacts to models/ folder
```

### Run Tests
```bash
pytest tests/ -v
```

## 🔐 Data Privacy

- Dataset is simulated telco customer data (Kaggle public dataset)
- No real customer PII included
- Safe for portfolio & GitHub



## 🙋 Author

**Aman Kumar Sharma**

Data Science | Machine Learning | Data Analytics

🔗 LinkedIn: https://www.linkedin.com/in/aman-kumar-sharma13/

🔗 GitHub: https://github.com/Aman13082001

---

## 🎯 Next Steps

1. **Customize** — Add your own data, retrain models, adjust thresholds
2. **Deploy** — Host on Streamlit Cloud, AWS, Heroku, or your preferred platform
3. **Extend** — Add more pages, implement real-time predictions, integrate with CRM
4. **Share** — Add to portfolio, show in interviews, demo to stakeholders

## 📞 Support

Found a bug? Have a feature request?  
Open an issue or reach out!

---

**Built with ❤️ using Streamlit**  
*All free and open source — no paid APIs, no cloud costs.*
