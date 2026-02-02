import pandas as pd

class AirlineCode:
    def __init__(self, data_path='data/DelayFlights-Airlines-mapping.csv'):
        self.data = pd.read_csv(data_path)
    
    def to_name(self, code: int):
        """
        Get airline code name from encoded value.
        """
        row = self.data[self.data['Encoded value'] == code]
        if not row.empty:
            return row['Airline Code'].values[0]
        return None
    
    def to_encoded(self, name: str):
        """
        Get encoded value from airline code name.
        """
        row = self.data[self.data['Airline Code'] == name]
        if not row.empty:
            return row['Encoded value'].values[0]
        return None
