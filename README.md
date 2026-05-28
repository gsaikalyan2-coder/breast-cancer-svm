\# 🔬 Breast Cancer SVM Classifier



A complete end-to-end Machine Learning pipeline for breast cancer diagnosis

built on the Wisconsin Diagnostic Breast Cancer dataset.



\## Live Demo

🚀 \[Try it on Hugging Face Spaces](https://huggingface.co/spaces/sk8069/breast-cancer-svm)



\## Project Overview

\- Model : Support Vector Machine (SVC) with RBF kernel

\- Dataset : Wisconsin Diagnostic Breast Cancer (569 samples, 30 features)

\- Accuracy : \~97% · AUC \~0.99

\- Task : Binary classification — Malignant vs Benign



\## Project Structure

breast-cancer-svm/

├── model/                    # Trained pipeline (not tracked by git)

├── backend/

│   ├── app.py                # FastAPI backend — /predict endpoint

│   └── requirements.txt

├── frontend/

│   └── streamlit\_app.py      # Streamlit UI (local — calls FastAPI)

├── hf\_space/

│   ├── app.py                # Standalone Streamlit for HF Spaces

│   └── requirements.txt

├── Dockerfile                # Docker container for FastAPI backend

└── README.md

## Pipeline Steps Completed

\- \[x] Data loading and EDA

\- \[x] Feature scaling with StandardScaler

\- \[x] sklearn Pipeline (scaler + SVC)

\- \[x] GridSearchCV hyperparameter tuning

\- \[x] Confusion matrix, ROC-AUC, cross-validation

\- \[x] 2D decision boundary visualization

\- \[x] SHAP feature importance

\- \[x] Model saved with joblib

\- \[x] Gradio demo (Google Colab)

\- \[x] FastAPI backend with /predict endpoint

\- \[x] Streamlit frontend

\- \[x] Docker containerization

\- \[x] Deployed to Hugging Face Spaces

\- \[x] GitHub versioning



\## Tech Stack

| Layer | Tool |

|-------|------|

| ML | scikit-learn, numpy, pandas |

| Explainability | SHAP |

| Backend | FastAPI, uvicorn |

| Frontend | Streamlit, Gradio |

| Container | Docker |

| Deployment | Hugging Face Spaces |

| Version control | GitHub |



\## Run Locally



\### Backend

```bash

cd backend

python -m uvicorn app: app --reload --port 8000

```



\### Frontend

```bash

cd frontend

python -m streamlit run streamlit\_app.py

```



\### Docker

```bash

docker build -t breast-cancer-svm.

docker run -p 8000:8000 breast-cancer-svm

```



\## Dataset

\[UCI Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))



\## Disclaimer

For educational purposes only — not a clinical diagnostic tool.

Always consult a qualified medical professional.

