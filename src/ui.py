import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.title("DataGenie Prediction")
date_from = st.date_input("Start Date", value=None)
date_to = st.date_input("End Date", value=None)
uploaded_file = st.file_uploader("Upload CSV", type="csv")
period = st.number_input("Prediction Period (0 for test)", min_value=0, value=0)

if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=["point_timestamp"], dayfirst=True)
    if len(data) < 5: st.error("Need at least 5 rows")
    else:
        if date_from and date_to:
            start, end = pd.to_datetime(date_from), pd.to_datetime(date_to)
            data = data[(data["point_timestamp"] >= start) & (data["point_timestamp"] <= end)]
        st.write(data.head())
        uploaded_file.seek(0)
        url = "http://localhost:8000/predict"
        params = {"period": period}
        if date_from: params["date_from"] = date_from.strftime("%d-%m-%Y")
        if date_to: params["date_to"] = date_to.strftime("%d-%m-%Y")
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        response = requests.post(url, files=files, params=params)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Best Model: {result['best_model']}")
            st.write(f"MAPE: {result['mape']:.4f}" if result['mape'] else "MAPE: N/A")
            if result.get("predictions"):
                predictions_df = pd.DataFrame(result["predictions"])
                predictions_df["point_timestamp"] = pd.to_datetime(predictions_df["point_timestamp"])
                st.write(predictions_df)
                plot_data = data.rename(columns={"point_value": "actual"})
                plot_data["predicted"] = None
                if not predictions_df.empty:
                    merged_data = pd.merge(plot_data[["point_timestamp", "actual"]], predictions_df, on="point_timestamp", how="left")
                    merged_data["predicted"] = merged_data["predicted"].fillna(merged_data["actual"])
                else: merged_data = plot_data
                all_dates = pd.date_range(plot_data["point_timestamp"].min(), predictions_df["point_timestamp"].max() if not predictions_df.empty else plot_data["point_timestamp"].max(), freq="D")
                merged_data = merged_data.set_index("point_timestamp").reindex(all_dates).reset_index().rename(columns={"index": "point_timestamp"})
                merged_data["actual"] = pd.to_numeric(merged_data["actual"], errors="coerce")
                merged_data["predicted"] = pd.to_numeric(merged_data["predicted"], errors="coerce")
                if "predicted" in merged_data:
                    fig = px.line(merged_data, x="point_timestamp", y=["actual", "predicted"], title="Actual vs Predicted")
                    fig.update_traces(selector=dict(name="actual"), line=dict(color="blue"), name="Actual")
                    fig.update_traces(selector=dict(name="predicted"), line=dict(color="red"), name="Predicted")
                    st.plotly_chart(fig)
        else: st.error(response.text)