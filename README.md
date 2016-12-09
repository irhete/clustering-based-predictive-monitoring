# clustering-based-predictive-monitoring
Clustering-based predictive process monitoring


## Usage

```python
from ClusteringPredictiveModel import ClusteringPredictiveModel

model = ClusteringPredictiveModel(case_id_col, event_col, label_col, timestamp_col, cat_cols, numeric_cols, n_clusters=n_clusters, n_estimators=100, random_state=22, fillna=True)
model.fit(train)

preds = model.predict_proba(test)
```

For more details see the jupyter notebook clustering_predictive_monitoring.ipynb
