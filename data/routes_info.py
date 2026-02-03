import pandas as pd
import json
import os

class RoutesInformation:
    def __init__(self, data_path='model/cnn_model_data'):
        with open(os.path.join(data_path, 'Flights_report_ids.json'), 'r') as f:
            data = json.load(f)
            self.airlines = pd.DataFrame(data['airline'].items(), columns=['Airline', 'Encoded value'])
        with open(os.path.join(data_path, 'Flights_routes.json'), 'r') as f:
            self.route = json.load(f)
    
    def to_airline_name(self, code: int):
        """
        Get airline ID from encoded value.
        """
        row = self.airlines[self.airlines['Encoded value'] == code]
        if not row.empty:
            return row['Airline'].values[0]
        return None
    
    def to_airline_encoded(self, name: str):
        """
        Get encoded value from airline ID.
        """
        row = self.airlines[self.airlines['Airline'] == name]
        if not row.empty:
            return row['Encoded value'].values[0]
        return None
    
    def route_info(self, origin: int, dest: int):
        info = self.route.get(f"{origin}_{dest}", {})
        return int(info.get("dist", 0)), int(info.get("dur", 0))
    
    def list_airlines(self):
        """
        List all airline names.
        """
        return self.airlines['Airline'].tolist()
    