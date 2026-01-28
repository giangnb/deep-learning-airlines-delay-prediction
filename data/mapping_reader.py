import pandas as pd
from sklearn.preprocessing import LabelEncoder

class MappingReader:
    def __init__(self, mapping_dir='mappings/'):
        self.mapping_dir = mapping_dir
        self.cache = {}
    
    def get_mapping(self, col_name):
        if col_name in self.cache:
            return self.cache[col_name]
        mapping = pd.read_csv(f"{self.mapping_dir}/{col_name}.csv")
        self.cache[col_name] = mapping
        return mapping
    
    def to_name(self, col_name, encoded_value):
        mapping = self.get_mapping(col_name)
        row = mapping[mapping['Encoded value'] == encoded_value]
        if not row.empty:
            return row['Category'].values[0]
        return None
    
    def to_encoded(self, col_name, category):
        mapping = self.get_mapping(col_name)
        row = mapping[mapping['Category'] == category]
        if not row.empty:
            return row['Encoded value'].values[0]
        return None
    
    def save_label_encoder(self, col_name, le: LabelEncoder):
        mapping = pd.DataFrame({
            'Category': le.classes_,
            'Encoded value': range(len(le.classes_))
        })
        mapping.to_csv(f"{self.mapping_dir}/{col_name}.csv", index=False)