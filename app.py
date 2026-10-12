import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Churn Risk Advisor",
    layout="wide"
)


RAW_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

PAYMENTS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
]


@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.joblib")

    with open("model_meta.json", "r") as file:
        metadata = json.load(file)

    return model, metadata


def prepare_input(raw, columns):
    encoded = pd.get_dummies(raw[RAW_COLS])

    return (
        encoded
        .reindex(columns=columns, fill_value=0)
        .astype(float)
    )


model, metadata = load_artifacts()
MODEL_COLUMNS = metadata["feature_columns"]


def score_customer(raw):
    features = prepare_input(
        raw,
        MODEL_COLUMNS
    )

    return model.predict_proba(
        features
    )[:, 1]


st.title("Customer Churn Risk Advisor")

st.caption(
    f"{metadata['model_name']} | "
    f"CV AUC: {metadata['cv_auc']:.3f} "
    f"+/- {metadata['cv_auc_std']:.3f} | "
    "Decision support only"
)


with st.sidebar:
    st.header("Customer Profile")

    tenure = st.slider(
        "Tenure in months",
        min_value=0,
        max_value=72,
        value=4
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly charges",
        min_value=18.0,
        max_value=120.0,
        value=85.0,
        step=0.5
    )

    internet_service = st.selectbox(
        "Internet service",
        [
            "Fiber optic",
            "DSL",
            "No"
        ]
    )

    selected_services = []

    if internet_service != "No":
        selected_services = st.multiselect(
            "Additional services",
            SERVICES,
            default=["StreamingTV"]
        )

    phone_service = st.radio(
        "Phone service",
        ["Yes", "No"],
        horizontal=True
    )

    multiple_lines = "No phone service"

    if phone_service == "Yes":
        multiple_lines = st.radio(
            "Multiple lines",
            ["No", "Yes"],
            horizontal=True
        )

    payment_method = st.selectbox(
        "Payment method",
        PAYMENTS
    )

    paperless_billing = st.radio(
        "Paperless billing",
        ["Yes", "No"],
        horizontal=True
    )

    st.subheader("Demographics")

    gender = st.radio(
        "Gender",
        ["Female", "Male"],
        horizontal=True
    )

    senior_citizen = st.checkbox(
        "Senior citizen"
    )

    partner = st.checkbox(
        "Has partner"
    )

    dependents = st.checkbox(
        "Has dependents"
    )

    threshold = st.slider(
        "Contact threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(metadata["threshold"]),
        step=0.05
    )


customer_row = {
    "gender": gender,
    "SeniorCitizen": int(senior_citizen),
    "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No",
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges
}


for service in SERVICES:
    if internet_service == "No":
        customer_row[service] = "No internet service"
    elif service in selected_services:
        customer_row[service] = "Yes"
    else:
        customer_row[service] = "No"


customer = pd.DataFrame([customer_row])

tab_one, tab_batch, tab_about = st.tabs(
    [
        "One Customer",
        "Batch Scoring",
        "About"
    ]
)


with tab_one:
    probability = float(
        score_customer(customer)[0]
    )

    if probability >= threshold:
        risk_band = "HIGH"
        action = "Contact now"
    elif probability >= threshold / 2:
        risk_band = "WATCH"
        action = "Review customer"
    else:
        risk_band = "LOW"
        action = "No immediate action"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churn probability",
        f"{probability:.1%}"
    )

    col2.metric(
        "Risk band",
        risk_band
    )

    col3.metric(
        "Recommended action",
        action
    )

    st.progress(
        min(probability, 1.0)
    )

    st.subheader("What could change the risk?")

    possible_changes = [
        ("Contract", "One year"),
        ("Contract", "Two year")
    ]

    possible_changes = [
        change for change in possible_changes
        if change[1] != contract
    ]

    if payment_method == "Electronic check":
        possible_changes.append(
            (
                "PaymentMethod",
                "Credit card (automatic)"
            )
        )

    rows = []

    for feature, value in possible_changes:
        alternative = customer.copy()
        alternative[feature] = value

        new_probability = float(
            score_customer(alternative)[0]
        )

        rows.append({
            "Change": f"{feature}: {value}",
            "New probability": round(
                new_probability,
                3
            ),
            "Difference": round(
                new_probability - probability,
                3
            )
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True
        )

    st.caption(
        "These are associations learned from the data, "
        "not guaranteed causal effects."
    )


with tab_batch:
    st.write(
        "Upload a CSV containing the original Telco customer columns."
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type="csv"
    )

    if uploaded_file is not None:
        batch_data = pd.read_csv(
            uploaded_file
        )

        missing_columns = [
            column for column in RAW_COLS
            if column not in batch_data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing columns: {missing_columns}"
            )
        else:
            batch_data["TotalCharges"] = pd.to_numeric(
                batch_data["TotalCharges"],
                errors="coerce"
            ).fillna(0)

            batch_data["p_churn"] = np.round(
                score_customer(batch_data),
                3
            )

            batch_data["contact"] = np.where(
                batch_data["p_churn"] >= threshold,
                "Yes",
                "No"
            )

            contacts = (
                batch_data["contact"] == "Yes"
            ).sum()

            st.write(
                f"{contacts} of "
                f"{len(batch_data)} customers "
                "are above the selected threshold."
            )

            st.dataframe(
                batch_data
                .sort_values(
                    "p_churn",
                    ascending=False
                )
                .head(50),
                hide_index=True
            )

            st.download_button(
                "Download scored CSV",
                batch_data.to_csv(index=False),
                "scored_customers.csv",
                "text/csv"
            )


with tab_about:
    st.subheader("Model information")

    visible_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "feature_columns"
    }

    st.json(visible_metadata)

    st.markdown(
        """
        **Limitations**

        This model was trained on one US telecom dataset from one period.
        It is intended to support retention prioritization. It should not be
        used for credit decisions, pricing decisions, or denying services.
        """
    )