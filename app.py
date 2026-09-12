import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from groq import Groq

st.set_page_config(
    page_title="AI Solar Recommendation",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ AI Solar Energy Recommendation System")
st.markdown(
    """
    **AI/ML-powered solar planning tool**

    This application estimates household electricity consumption,
    predicts future energy demand, estimates solar generation,
    compares different PV system sizes, and provides a personalized
    solar recommendation.
    """
)
st.divider()


def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def create_sample_consumption_data(monthly_consumption):
    np.random.seed(42)

    months = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=24,
        freq="MS"
    )

    values = []

    for month in months:
        if month.month in [6, 7, 8]:
            seasonal_factor = 1.20
        elif month.month in [12, 1, 2]:
            seasonal_factor = 1.10
        elif month.month in [4, 5, 9]:
            seasonal_factor = 1.05
        else:
            seasonal_factor = 0.90

        noise = np.random.normal(0, 0.04)
        consumption = monthly_consumption * seasonal_factor * (1 + noise)
        values.append(max(consumption, 50))

    return pd.DataFrame({
        "date": months,
        "consumption_kwh": values
    })


def train_consumption_model(df):
    data = df.copy()
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

    features = ["month", "year", "month_sin", "month_cos"]

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        max_depth=6
    )
    model.fit(data[features], data["consumption_kwh"])
    return model


def predict_next_12_months(model):
    future_dates = pd.date_range(
        start=pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1),
        periods=12,
        freq="MS"
    )

    future = pd.DataFrame({"date": future_dates})
    future["month"] = future["date"].dt.month
    future["year"] = future["date"].dt.year
    future["month_sin"] = np.sin(2 * np.pi * future["month"] / 12)
    future["month_cos"] = np.cos(2 * np.pi * future["month"] / 12)

    features = ["month", "year", "month_sin", "month_cos"]
    future["predicted_consumption_kwh"] = model.predict(future[features])

    return future


def estimate_solar_generation(
    solar_size_kw,
    peak_sun_hours,
    performance_ratio=0.80
):
    daily_generation = solar_size_kw * peak_sun_hours * performance_ratio
    monthly_generation = daily_generation * 30
    return daily_generation, monthly_generation


def recommend_solar_size(
    predicted_monthly_consumption,
    peak_sun_hours,
    roof_area,
    budget,
    battery_required
):
    solar_sizes = np.arange(2, 11, 1)
    results = []
    estimated_cost_per_kw = 180000

    for size in solar_sizes:
        daily_generation, monthly_generation = estimate_solar_generation(
            size, peak_sun_hours
        )

        coverage = (
            monthly_generation / predicted_monthly_consumption
        ) * 100

        estimated_cost = size * estimated_cost_per_kw
        required_roof_area = size * 60

        score = max(0, 100 - abs(100 - coverage))

        if required_roof_area <= roof_area:
            score += 30
        else:
            score -= 50

        if estimated_cost <= budget:
            score += 30
        else:
            score -= 50

        if battery_required:
            score += 5

        results.append({
            "Solar Size (kW)": size,
            "Daily Generation (kWh)": daily_generation,
            "Monthly Generation (kWh)": monthly_generation,
            "Solar Coverage (%)": coverage,
            "Estimated Cost (PKR)": estimated_cost,
            "Required Roof Area (sq ft)": required_roof_area,
            "Score": score
        })

    results_df = pd.DataFrame(results)
    recommendation = results_df.loc[results_df["Score"].idxmax()]

    return recommendation, results_df


def generate_ai_explanation(
    location,
    monthly_consumption,
    predicted_consumption,
    solar_size,
    solar_generation,
    coverage,
    roof_area,
    budget,
    battery_required
):
    client = get_groq_client()

    if client is None:
        return (
            "Groq API key is not configured. The numerical recommendation "
            "is still available. Add GROQ_API_KEY to Streamlit Secrets "
            "to enable the AI explanation."
        )

    prompt = f"""
You are an expert solar energy advisor.

Explain this solar recommendation to a homeowner in simple,
easy-to-understand English.

Home information:
- Location: {location}
- Current monthly electricity consumption: {monthly_consumption:.0f} kWh
- Predicted monthly consumption: {predicted_consumption:.0f} kWh
- Recommended solar size: {solar_size:.0f} kW
- Estimated monthly solar generation: {solar_generation:.0f} kWh
- Estimated solar coverage: {coverage:.1f}%
- Roof area: {roof_area:.0f} sq ft
- Budget: {budget:,.0f} PKR
- Battery required: {battery_required}

Explain:
1. Why this solar size was recommended.
2. What the predicted consumption means.
3. What the estimated solar generation means.
4. What solar coverage means.
5. Whether a battery may be useful.
6. Mention that actual generation depends on weather, shading,
   panel orientation, equipment efficiency, and site conditions.

Do not claim that these are guaranteed savings or exact results.
Keep the explanation professional and concise.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional solar energy advisor."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Unable to generate Groq explanation: {str(e)}"


st.sidebar.header("🏠 Home Information")

location = st.sidebar.text_input(
    "Location",
    value="Islamabad, Pakistan"
)

monthly_consumption = st.sidebar.number_input(
    "Current Monthly Consumption (kWh)",
    min_value=50.0,
    max_value=5000.0,
    value=700.0,
    step=50.0
)

roof_area = st.sidebar.number_input(
    "Available Roof Area (sq ft)",
    min_value=100.0,
    max_value=10000.0,
    value=500.0,
    step=50.0
)

budget = st.sidebar.number_input(
    "Solar Budget (PKR)",
    min_value=100000.0,
    max_value=10000000.0,
    value=1000000.0,
    step=50000.0
)

peak_sun_hours = st.sidebar.slider(
    "Average Peak Sun Hours",
    min_value=3.0,
    max_value=7.0,
    value=5.0,
    step=0.1
)

battery_required = st.sidebar.checkbox("I need battery backup")

st.sidebar.divider()

run_analysis = st.sidebar.button(
    "☀️ Analyze Solar Requirement",
    use_container_width=True
)

if run_analysis:
    with st.spinner("Preparing electricity consumption data..."):
        consumption_df = create_sample_consumption_data(monthly_consumption)

    with st.spinner("Training AI/ML consumption forecasting model..."):
        model = train_consumption_model(consumption_df)

    with st.spinner("Predicting future electricity consumption..."):
        forecast_df = predict_next_12_months(model)

    predicted_monthly_consumption = (
        forecast_df["predicted_consumption_kwh"].mean()
    )

    with st.spinner("Comparing solar system sizes..."):
        recommendation, comparison_df = recommend_solar_size(
            predicted_monthly_consumption,
            peak_sun_hours,
            roof_area,
            budget,
            battery_required
        )

    recommended_size = recommendation["Solar Size (kW)"]
    monthly_generation = recommendation["Monthly Generation (kWh)"]
    daily_generation = recommendation["Daily Generation (kWh)"]
    coverage = recommendation["Solar Coverage (%)"]
    estimated_cost = recommendation["Estimated Cost (PKR)"]

    st.subheader("📊 Solar Recommendation")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Recommended Solar", f"{recommended_size:.0f} kW")
    col2.metric(
        "Predicted Consumption",
        f"{predicted_monthly_consumption:.0f} kWh/month"
    )
    col3.metric(
        "Solar Generation",
        f"{monthly_generation:.0f} kWh/month"
    )
    col4.metric("Solar Coverage", f"{coverage:.1f}%")

    st.divider()

    st.subheader("📈 AI/ML Electricity Consumption Forecast")

    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=consumption_df["date"],
            y=consumption_df["consumption_kwh"],
            mode="lines+markers",
            name="Historical Consumption"
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_consumption_kwh"],
            mode="lines+markers",
            name="Predicted Consumption"
        )
    )

    fig1.update_layout(
        xaxis_title="Month",
        yaxis_title="Electricity Consumption (kWh)",
        hovermode="x unified"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("☀️ Solar System Comparison")

    display_df = comparison_df.copy()

    display_df["Daily Generation (kWh)"] = (
        display_df["Daily Generation (kWh)"].round(1)
    )

    display_df["Monthly Generation (kWh)"] = (
        display_df["Monthly Generation (kWh)"].round(0)
    )

    display_df["Solar Coverage (%)"] = (
        display_df["Solar Coverage (%)"].round(1)
    )

    display_df["Estimated Cost (PKR)"] = (
        display_df["Estimated Cost (PKR)"].round(0).astype(int)
    )

    st.dataframe(
        display_df[
            [
                "Solar Size (kW)",
                "Daily Generation (kWh)",
                "Monthly Generation (kWh)",
                "Solar Coverage (%)",
                "Estimated Cost (PKR)",
                "Required Roof Area (sq ft)"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            x=comparison_df["Solar Size (kW)"],
            y=comparison_df["Monthly Generation (kWh)"],
            name="Solar Generation"
        )
    )

    fig2.add_hline(
        y=predicted_monthly_consumption,
        line_dash="dash",
        annotation_text="Predicted Monthly Consumption"
    )

    fig2.update_layout(
        xaxis_title="Solar System Size (kW)",
        yaxis_title="Estimated Generation (kWh/month)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("💡 Recommended System Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Recommended Solar Size:** {recommended_size:.0f} kW")
        st.write(f"**Estimated Daily Generation:** {daily_generation:.1f} kWh")
        st.write(
            f"**Estimated Monthly Generation:** "
            f"{monthly_generation:.0f} kWh"
        )

    with col2:
        st.write(
            f"**Predicted Monthly Consumption:** "
            f"{predicted_monthly_consumption:.0f} kWh"
        )
        st.write(f"**Estimated Solar Coverage:** {coverage:.1f}%")
        st.write(f"**Estimated System Cost:** {estimated_cost:,.0f} PKR")

    st.subheader("🔋 Battery Assessment")

    if battery_required:
        st.info(
            "You selected battery backup. A battery can be useful for "
            "storing daytime solar energy and supplying loads when solar "
            "production is low. Exact battery sizing should be based on "
            "backup load and required backup hours."
        )
    else:
        st.info(
            "Battery backup was not selected. If your main objective is "
            "reducing grid electricity consumption, a battery may not be "
            "necessary depending on your local grid/export arrangement."
        )

    st.subheader("🤖 AI Explanation — Powered by Groq")

    with st.spinner("Generating personalized AI explanation..."):
        explanation = generate_ai_explanation(
            location,
            monthly_consumption,
            predicted_monthly_consumption,
            recommended_size,
            monthly_generation,
            coverage,
            roof_area,
            budget,
            battery_required
        )

    st.markdown(explanation)

    st.divider()

    st.warning(
        """
        ⚠️ **Important:** This is a demonstration/prototype tool.
        Solar generation and cost estimates are approximate. Actual system
        sizing should consider site-specific solar irradiance, shading,
        roof orientation, panel specifications, inverter efficiency,
        wiring losses, local electricity tariffs, export/net-metering
        rules, and professional site assessment.
        """
    )

else:
    st.info(
        "👈 Enter your home information in the sidebar and click "
        "**Analyze Solar Requirement** to start."
    )

    st.markdown(
        """
        ### What this application demonstrates

        **1. Consumption Forecasting**

        Uses a machine-learning model to estimate future household
        electricity consumption.

        **2. Solar Generation Estimation**

        Estimates how much energy different solar system sizes could produce.

        **3. Personalized Recommendation**

        Compares system sizes using predicted consumption, roof area,
        budget, and battery preference.

        **4. AI Explanation**

        Groq provides a natural-language explanation of the recommendation.
        """
    )
