from data.airport_code import AirportCode
from data.routes_info import RoutesInformation
from application.models_utilities import ModelUtilities
import numpy as np
from datetime import datetime

def initialize_models(artifacts_path="/model/training_artifacts.pkl"):
    model_utils = ModelUtilities()
    try:
        model_utils.load_keras(
        name="Multi-Model",
        path="model/MultiModels-20260201-041816.keras",
        artifacts_path=artifacts_path
    )
    except Exception as e:
        print(f"Failed loading model Multi-Model: {e}")
    try:
        model_utils.load_keras(
            name="TabTransformer",
            path="model/Transformer-20260202-084654-NEW DAT-5epochs-huber-scaled targets.h5",   
            artifacts_path=artifacts_path,
            model_type="TabTransformer"
        )
    except Exception as e:
        print(f"Failed loading model TabTransformer: {e}")
    try:
        model_utils.load_cnn_model(
            name="Convolutional NN",
            path="model/CNN-model_classifier.keras",
            id_map_path="model/cnn_model_data/Flights_report_ids.json",
            history_data_path="model/cnn_model_data/Flights_history.json",
            route_map_path="model/cnn_model_data/Flights_routes.json"
        )
    except Exception as e:
        print(f"Failed loading model Convolutional NN: {e}")
    
    print(f"Initialized models: {model_utils.get_model_names()}")
    return model_utils

def main():
    airport_code = AirportCode()
    #airlines_code = RoutesInformation('model/cnn_model_data/Flights_report_ids.json')
    adaptor = initialize_models("model/training_artifacts.pkl")

    origin = input("Origin Airport Code: ")
    dest = input("Destination Airport Code: ")

    origin = airport_code.to_encoded(origin.upper())
    dest = airport_code.to_encoded(dest.upper())

    distance = int(input("Distance Miles: "))

    date = input("Date (YYYY-MM-DD): ")
    date = datetime.strptime(date, "%Y-%m-%d")

    all_result = adaptor.get_all_models_predictions({
        "Airline": input("Airline Code: ").upper(),
        "Origin Airport Code": origin,
        "Destination Airport Code": dest,
        "Departure Block Hour": int(input("Departure Hour (0-23): ")),
        "Day Of Week": date.isoweekday(),
        "Month": date.month,
        "Fly Time Scheduled": int(input("Fly time (minutes): ")),
        "Distance Miles": distance,
        "Distance Group": 1 if distance <= 250 else 2 if distance <= 750 else 3
    })

    for model_name, result in all_result.items():
        print(f"\n=== Model: {model_name} ===")
        print_result(result)
        print("\n" + "="*50 + "\n")

def print_result(result):
    if result is None:
        print("Prediction failed.")
        return
    try:
        print(f"Delay probability: {result['probability_of_delay'] * 100:.2f}%")
    except Exception:
        pass
    try:
        print(f"Primary reason: {result['primary_reason']}")
    except Exception:
        pass
    if 'predicted_delays_minutes' in result:
        print(f"Predicted Delays (minutes):")
        for reason, minutes in result['predicted_delays_minutes'].items():
            try:
                print(f" - {reason}: {np.expm1(minutes):.2f} \t mins")
            except Exception:
                pass

if __name__ == "__main__":
    main()
