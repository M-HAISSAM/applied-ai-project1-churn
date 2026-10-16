# Churn Risk Advisor

Streamlit project deployed on Streamlit Cloud from this GitHub repo (`main` branch).

## Files
- `app.py`: churn dashboard (XGBoost model in `churn_model.joblib`)
- `dataset_lab.py`: "Dataset Lab": upload any CSV/Excel, EDA, trains 8 ML algorithms, clustering
- `theme.py`: dark theme CSS + chart helpers
- `.streamlit/config.toml`: theme config

## Rules
1. Use this Python only: `E:\claude_test_venv\Scripts\python.exe` (C: drive is almost full, never pip install there).
2. Run the app with: `E:\claude_test_venv\Scripts\python.exe -m streamlit run app.py`
3. Keep `xgboost>=3.2,<3.4` and `scikit-learn==1.6.1` in `requirements.txt` (3.1.0 rejects the model's feature names, 3.4.x can't load the model).
4. After every change, test headlessly with `streamlit.testing.v1.AppTest`: both workspaces (Churn Risk Advisor and Dataset Lab), and train models on at least one classification and one regression demo dataset. Report any exceptions.
5. Use `width='stretch'`, not `use_container_width`.
6. Never commit `__pycache__`.
7. Before git commit/push, show me the changed files and wait for my OK.
8. Follow THEME.md ("Midnight Mono") for all UI: import colour tokens from theme.py, never hardcode hex colours in app.py or dataset_lab.py. Keep theme.py, .streamlit/config.toml and THEME.md in sync.
