import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

from data.airport_code import AirportCode
from data.routes_info import RoutesInformation
from application.models_utilities import ModelUtilities

st.set_page_config(layout="wide",
                   page_title="Fligth Delay Prediction",
                   page_icon="✈️")

__cached__obj = {}

# --- MODEL & DATA LOADING ---
# Use caching for resource-intensive initializations to run only once.
def initialize_models(artifacts_path="model/training_artifacts.pkl"):
    """Loads all predictive models from disk."""
    if "model_utils" in __cached__obj:
        return __cached__obj["model_utils"]
    
    model_utils = ModelUtilities()
    # Error handling for each model to ensure app runs even if one fails
    try:
        model_utils.load_keras(
            name="Multi-Model",
            path="model/MultiModels-20260201-041816.keras",
            artifacts_path=artifacts_path
        )
    except Exception as e:
        st.error(f"Failed loading model Multi-Model: {e}")
    try:
        model_utils.load_keras(
            name="TabTransformer",
            path="model/Transformer-20260202-084654-NEW DAT-5epochs-huber-scaled targets.h5",
            artifacts_path=artifacts_path,
            model_type="TabTransformer"
        )
    except Exception as e:
        st.error(f"Failed loading model TabTransformer: {e}")
    try:
        model_utils.load_cnn_model(
            name="Convolutional NN",
            path="model/CNN-model_classifier.keras",
            id_map_path="model/cnn_model_data/Flights_report_ids.json",
            history_data_path="model/cnn_model_data/Flights_history.json",
            route_map_path="model/cnn_model_data/Flights_routes.json"
        )
    except Exception as e:
        st.error(f"Failed loading model Convolutional NN: {e}")
    
    __cached__obj["model_utils"] = model_utils
    return model_utils

def get_airport_handler():
    """Loads airport code mapping."""
    if "airport_code" in __cached__obj:
        return __cached__obj["airport_code"]
    __cached__obj["airport_code"] = AirportCode()
    return __cached__obj["airport_code"]

def get_routes_handler():
    """Loads airline and route information."""
    if "routes_info" in __cached__obj:
        return __cached__obj["routes_info"]
    __cached__obj["routes_info"] = RoutesInformation(data_path='model/cnn_model_data')
    return __cached__obj["routes_info"]

def get_distance_group(distance):
    dist = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    for i in range(len(dist)):
        if distance < dist[i]:
            return i + 1, f"{'Less than' if i<=0 else (str(dist[i-1]) + ' to')} {dist[i]} miles"
        elif i == len(dist) - 1:
            return i + 2, f"Equal or greater than {dist[i]} miles"

# --- CALLBACKS ---
def update_route_info():
    """
    Callback function to automatically update distance and fly time
    when origin or destination airport changes.
    """
    origin_name = st.session_state.origin_airport
    dest_name = st.session_state.dest_airport

    if not origin_name or not dest_name:
        return
    
    airport_code_handler = get_airport_handler()
    routes_info_handler = get_routes_handler()
    
    origin_code = airport_code_handler.to_encoded(origin_name)
    dest_code = airport_code_handler.to_encoded(dest_name)
    
    if origin_code is not None and dest_code is not None and origin_code != dest_code:
        dist, dur = routes_info_handler.route_info(origin_code, dest_code)
        # Only update if the route is found in the lookup
        if dist > 0 and dur > 0:
            st.session_state.distance_miles = dist
            st.session_state.fly_time = dur
            _, distance_name = get_distance_group(dist)
            st.session_state.distance_group_txt = distance_name

# --- INITIALIZATION ---
st.title("🛫 Flight Delay Detection")
st.markdown("\n")

# Load all necessary resources
adaptor = initialize_models()
airport_code = get_airport_handler()
routes_info = get_routes_handler()

# Get lists for dropdowns
airport_list = sorted(airport_code.list_airports())
airline_list = sorted(routes_info.list_airlines())

# Initialize session state for form inputs if they don't exist
if "distance_miles" not in st.session_state:
    st.session_state.distance_miles = 2475 # JFK to LAX default
if "fly_time" not in st.session_state:
    st.session_state.fly_time = 335 # JFK to LAX default
if "btn_disabled" not in st.session_state:
    st.session_state.btn_disabled = False

def dummy():
    pass

# --- UI FORM ---
with st.container(border=True):
    st.markdown("## Flight Details")
    col1, col2 = st.columns(2)

    with col1:
        origin_airport_name = st.selectbox(
            "Origin airport",
            options=airport_list,
            key="origin_airport",
            index=airport_list.index("JFK") if "JFK" in airport_list else 0,
            on_change=update_route_info
        )
        fly_date_val = st.date_input(
            "Fly date",
            min_value=date(2025, 7, 31),
            value=datetime.now().date()
        )
        airline_name = st.selectbox(
            "Airline",
            options=airline_list,
            index=airline_list.index("AA") if "AA" in airline_list else 0,
        )
        fly_time_minutes = st.number_input(
            "Fly time (minutes)",
            min_value=15,
            max_value=1500,
            key="fly_time"
        )

    with col2:
        dest_airport_name = st.selectbox(
            "Destination airport",
            options=airport_list,
            key="dest_airport",
            index=airport_list.index("LAX") if "LAX" in airport_list else 0,
            on_change=update_route_info
        )
        departure_hour = st.number_input(
            "Departure hour",
            min_value=0,
            max_value=23,
            value=17
        )
        distance_miles = st.number_input(
            "Distance (miles)",
            min_value=10,
            max_value=12000,
            key="distance_miles"
        )
        distance_group_txt = st.text_input(
            "Distance group",
            value="2250 to  2500 miles" if 'distance_group_txt' not in st.session_state else st.session_state.distance_group_txt,
            disabled=True,
            key="distance_group_txt"
        )

    st.divider()

    btn_calculate = st.button("**Calculate Delay**", type="primary", width="stretch", disabled=st.session_state.btn_disabled)
    st.markdown("\n")

# --- FORM SUBMISSION LOGIC ---
if btn_calculate:
    with st.container(border=True):
        if origin_airport_name == dest_airport_name:
            st.error("Origin and Destination airports cannot be the same.")
        else:
            # Prepare input data for models
            origin_enc = airport_code.to_encoded(origin_airport_name)
            dest_enc = airport_code.to_encoded(dest_airport_name)
            distance = int(distance_miles)
            distance_group, distance_name = get_distance_group(distance)

            input_data = {
                "Airline": airline_name,
                "Origin Airport Code": origin_enc,
                "Destination Airport Code": dest_enc,
                "Departure Block Hour": int(departure_hour),
                "Day Of Week": fly_date_val.isoweekday(),
                "Month": fly_date_val.month,
                "Fly Time Scheduled": int(fly_time_minutes),
                "Distance Miles": distance,
                "Distance Group": distance_group
            }

            # Get predictions from all loaded models
            with st.spinner("Calculating predictions from all models..."):
                st.session_state.btn_disabled = True
                all_result = adaptor.get_all_models_predictions(input_data)
                st.session_state.btn_disabled = False
                if "all_result" not in st.session_state:
                    st.session_state.all_result = all_result

            # Display result
            st.markdown(f"## Delay Prediction {st.session_state.origin_airport} → {st.session_state.dest_airport}")

            # Display results in accordion elements
            for model_name, result in all_result.items():
                with st.expander(f"Results from **{model_name}**", expanded=True):
                    if result is None:
                        st.warning(f"Prediction failed or is not available for model {model_name}.")
                        continue
                    
                    res_col1, res_col2 = st.columns([1,3])
                    
                    with res_col1:
                        if result.get("primary_reason"):
                            st.metric(label="Primary Reason for Delay", value=str(result["primary_reason"]))
                        st.markdown("\n")
                        if result.get("probability_of_delay") is not None:
                            prob_value = result['probability_of_delay']
                            st.metric(label="Probability of Delay", value=f"{prob_value * 100:.2f}%")
                    
                    with res_col2:
                        if 'predicted_delays_minutes' in result and result['predicted_delays_minutes']:
                            st.write("Predicted Delay Breakdown (minutes):")
                            content = pd.DataFrame([{"Reason": k, "Delay": round(np.expm1(v), 2)} for k, v in result['predicted_delays_minutes'].items() if v is not None and v > 0])
                            st.dataframe(content.sort_values(by='Delay', ascending=False), hide_index=True, width='stretch')
            
        st.session_state.btn_disabled = False

# Footer
st.divider()
if "all_result" not in st.session_state:
    st.info(f"Available models: **{', '.join(initialize_models().get_model_names())}**")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: .8em;'>
        Developed by <b>Group 1</b> &copy; 2026 <br/>
        AASD 4010 &bull; George Brown Polytecnic 
    </div>
    """,
    unsafe_allow_html=True
)