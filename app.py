# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="shell brand-row">
      <div style="display:flex;align-items:center;">
        <div class="brand-mark">CR</div>
        <div><div class="brand-title">Churn Risk Advisor</div><div class="brand-subtitle">Customer retention decision support</div></div>
      </div>
      <div class="status-pill">● &nbsp;Production model online</div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not artifact_ready:
    st.error("The application loaded, but the model artifact could not be opened. Confirm that churn_model.joblib and model_meta.json are in the repository root.")
    st.stop()


if page == "Customer assessment":
    st.markdown('<div class="section-title">Customer assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Enter the customer profile to estimate churn risk and support the next retention action.</div>', unsafe_allow_html=True)

    with st.form("customer_form"):
        st.markdown("#### Customer profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            partner = st.selectbox("Partner", ["Yes", "No"])
        with c2:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            phone = st.selectbox("Phone service", ["Yes", "No"])
        with c3:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        with c4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        st.markdown("#### Services and billing")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        with s2:
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with s3:
            tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        with s4:
            stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        b1, b2 = st.columns(2)
        with b1:
            stream_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        with b2:
            monthly = st.number_input("Monthly charges (PKR)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)

        submitted = st.form_submit_button("Assess customer", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech, "StreamingTV": stream_tv, "StreamingMovies": stream_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": float(tenure) * float(monthly),
        }])
        probability = float(predict_probability(row, bundle)[0])
        predicted = probability >= selected_threshold
        if probability >= max(selected_threshold, .70):
            level, title, css = "High", "Prioritise retention outreach", "risk-high"
            copy = "This profile is materially above the selected decision threshold. Review the account and consider a timely retention offer."
        elif probability >= selected_threshold:
            level, title, css = "Moderate", "Review for targeted outreach", "risk-medium"
            copy = "The customer is above the decision threshold. Use account context and recent interactions before selecting an offer."
        else:
            level, title, css = "Low", "Monitor through normal service", "risk-low"
            copy = "The estimated risk is below the decision threshold. No special intervention is indicated by this model alone."

        st.markdown("<div class='section-title' style='margin-top:1.65rem;'>Assessment result</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-panel {css}'><div class='result-label'>Recommended disposition · {level} risk</div><div class='result-title'>{title}</div><div class='result-copy'>{copy}</div></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated churn probability", f"{probability:.1%}")
        m2.metric("Decision threshold", f"{selected_threshold:.1%}")
        m3.metric("Model decision", "Outreach" if predicted else "Monitor")
        m4.metric("Potential avoided loss", fmt_money(6000 if predicted else 0))
        st.caption("The probability is a model estimate, not a guarantee. The threshold is configured from the business-cost analysis in the project notebook.")


elif page == "Batch scoring":
    st.markdown('<div class="section-title">Batch scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Upload a CSV with customer records using the same fields as the training data. The scored file can be downloaded for review.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Customer CSV", type=["csv"], help="Required columns: the customer profile fields used by the model.")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [col for col in RAW_COLS if col not in data.columns]
        if missing:
            st.error(f"The file is missing {len(missing)} required column(s): {', '.join(missing)}")
        else:
            probabilities = predict_probability(data[RAW_COLS], bundle)
            scored = data.copy()
            scored["churn_probability"] = probabilities
            scored["recommended_action"] = np.where(probabilities >= selected_threshold, "Review for outreach", "Monitor")
            st.success(f"Scored {len(scored):,} customer records.")
            a, b, c = st.columns(3)
            a.metric("Records", f"{len(scored):,}")
            b.metric("Above threshold", f"{(probabilities >= selected_threshold).sum():,}")
            c.metric("Average risk", f"{probabilities.mean():.1%}")
            st.dataframe(scored.head(100), use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "churn_scored_customers.csv", "text/csv", type="primary")


else:
    st.markdown('<div class="section-title">Model details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Reference information for reviewers and the retention team.</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.metric("Decision threshold", f"{selected_threshold:.1%}")
    d2.metric("Estimator", type(model_object).__name__)
    d3.metric("Input fields", len(RAW_COLS))
    st.markdown("#### Model card")
    if metadata:
        pretty = {str(k).replace("_", " ").title(): v for k, v in metadata.items() if not isinstance(v, (dict, list))}
