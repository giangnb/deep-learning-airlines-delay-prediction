
from application.flight_delay_predictor import FlightDelayPredictor
from application.cnn_model_classifier_pre import DelayCausePredictor

class ModelUtilities:
    __cached__models = {}

    def __init__(self):
        pass

    def load_keras(self, name: str, path: str, **kwargs) -> FlightDelayPredictor:
        """
        Load and cache a Keras model.
        """
        if name not in self.__cached__models:
            model = FlightDelayPredictor(model_path=path, 
                                         artifacts_path=kwargs.get("artifacts_path", "/model/training_artifacts.pkl"), 
                                         model_type=kwargs.get("model_type", "ANN"))
            self.__cached__models[name] = model
        return self.__cached__models[name]
    
    def load_cnn_model(self, name: str, path: str, **kwargs) -> DelayCausePredictor:
        """
        Load and cache a CNN Keras model.
        """
        if name not in self.__cached__models:
            model = DelayCausePredictor(
                model_path=path,
                id_map_path=kwargs.get("id_map_path", "/model/cnn_model_data/Flights_report_ids.json"),
                history_data_path=kwargs.get("history_data_path", "/model/cnn_model_data/Flights_history.json"),
                route_map_path=kwargs.get("route_map_path", "/model/cnn_model_data/Flights_routes.json")
            )
            self.__cached__models[name] = model
        return self.__cached__models[name]
    
    def get_prediction(self, model_name: str, **kwargs: dict):
        """
        Get prediction from a cached model.
        """
        if model_name not in self.__cached__models:
            raise ValueError(f"Model '{model_name}' is not loaded.")
        model = self.__cached__models[model_name]
        if isinstance(model, FlightDelayPredictor):
            return model.predict(kwargs)
        elif isinstance(model, DelayCausePredictor):
            return model.predict(kwargs.get("Airline", 1), kwargs.get("Origin Airport Code"), 
                                 kwargs.get("Destination Airport Code"), kwargs.get("Month"), **kwargs)
        else:
            raise ValueError(f"Unknown model type for '{model_name}'.")
        
    def get_all_models_predictions(self, input_data: dict):
        """
        Get predictions from all cached models.
        """
        results = {}
        for name, model in self.__cached__models.items():
            try:
                results[name] = self.get_prediction(name, **input_data)
            except Exception as e:
                results[name] = None
                print(f"Inference error for '{name}': {e}")
                raise
        return results
    
    def get_model_names(self):
        """
        Get names of all cached models.
        """
        return list(self.__cached__models.keys())