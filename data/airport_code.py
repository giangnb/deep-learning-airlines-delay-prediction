import pandas as pd

class AirportCode:
    def __init__(self, data_path='data/DelayFlights-Airports-mapping.csv'):
        self.data = pd.read_csv(data_path)
    
    def to_name(self, code: int):
        """
        Get airport name from encoded value.
        """
        row = self.data[self.data['Encoded value'] == code]
        if not row.empty:
            return row['Airport ID'].values[0]
        return None
    
    def to_encoded(self, name: str):
        """
        Get encoded value from airport name.
        """
        row = self.data[self.data['Airport ID'] == name]
        if not row.empty:
            return row['Encoded value'].values[0]
        return None
