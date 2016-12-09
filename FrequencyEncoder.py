from sklearn.base import TransformerMixin
import pandas as pd

class FrequencyEncoder(TransformerMixin):
    
    def __init__(self, case_id_col, event_col):
        self.case_id_col = case_id_col
        self.event_col = event_col
        self.columns = None
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, data, y=None):
        
        # reshape: each activity will be separate column
        data_events = pd.get_dummies(data[self.event_col])
        data_events[self.case_id_col] = data[self.case_id_col]
        
        # add missing columns if necessary
        if self.columns is None:
            self.columns = data_events.columns
        else:
            missing_cols = [col for col in self.columns if col not in data_events.columns]
            for col in missing_cols:
                data_events[col] = 0
            data_events = data_events[self.columns]
        
        # aggregate activities by case
        grouped = data_events.groupby(self.case_id_col)
        return grouped.sum()