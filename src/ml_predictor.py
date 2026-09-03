import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

class DisruptionPredictor:
    def __init__(self, model_path="data/processed/disruption_model.joblib"):
        self.model_path = model_path
        self.closure_clf = None
        self.landslide_reg = None
        self.flood_reg = None
        self.feature_names = [
            "rainfall_intensity_mm",
            "rain_accum_24h_mm",
            "slope_deg",
            "elevation_m",
            "distance_to_river_m",
            "historical_landslide_count",
            "road_condition_index"
        ]
        
        self.feature_display_names = {
            "rainfall_intensity_mm": "Rainfall (Current)",
            "rain_accum_24h_mm": "24h Saturation",
            "slope_deg": "Terrain Slope",
            "elevation_m": "Elevation",
            "distance_to_river_m": "River Proximity",
            "historical_landslide_count": "Historical Incidents",
            "road_condition_index": "Road Quality"
        }
        
        self._load_or_train()

    def _generate_synthetic_data(self, n_samples=1000):
        # Synthetic data generation based on Meghalaya/Assam terrain characteristics
        rs = np.random.RandomState(42)
        
        rainfall_intensity_mm = rs.uniform(0, 100, n_samples)
        rain_accum_24h_mm = rainfall_intensity_mm * rs.uniform(2, 6, n_samples) + rs.uniform(0, 50, n_samples)
        slope_deg = rs.uniform(0, 45, n_samples)
        elevation_m = rs.uniform(50, 2000, n_samples)
        distance_to_river_m = rs.uniform(10, 5000, n_samples)
        historical_landslide_count = rs.poisson(lam=1.5, size=n_samples)
        road_condition_index = rs.uniform(0, 1, n_samples) # 1.0 is perfect, 0.0 is terrible
        
        # Risk formulas for synthetic targets
        landslide_risk_score = (
            (slope_deg / 45) * 0.4 +
            (rain_accum_24h_mm / 400) * 0.3 +
            (historical_landslide_count / 10) * 0.2 +
            (1.0 - road_condition_index) * 0.1
        )
        
        flood_risk_score = (
            np.clip(1.0 - (distance_to_river_m / 1000), 0, 1) * 0.4 +
            (rain_accum_24h_mm / 300) * 0.3 +
            np.clip(1.0 - (elevation_m / 500), 0, 1) * 0.3
        )
        
        # Add some noise
        landslide_prob = np.clip(landslide_risk_score + rs.normal(0, 0.05, n_samples), 0, 1)
        flood_prob = np.clip(flood_risk_score + rs.normal(0, 0.05, n_samples), 0, 1)
        
        closure_prob = np.clip(np.maximum(landslide_prob, flood_prob) * 1.2, 0, 1)
        binary_closure = (closure_prob > 0.65).astype(int)
        
        df = pd.DataFrame({
            "rainfall_intensity_mm": rainfall_intensity_mm,
            "rain_accum_24h_mm": rain_accum_24h_mm,
            "slope_deg": slope_deg,
            "elevation_m": elevation_m,
            "distance_to_river_m": distance_to_river_m,
            "historical_landslide_count": historical_landslide_count,
            "road_condition_index": road_condition_index,
            "landslide_prob": landslide_prob,
            "flood_prob": flood_prob,
            "binary_closure": binary_closure
        })
        
        return df

    def _load_or_train(self):
        # We enforce fast in-memory train since the dataset is just 1000 records
        df = self._generate_synthetic_data()
        X = df[self.feature_names]
        
        self.closure_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.closure_clf.fit(X, df["binary_closure"])
        
        self.landslide_reg = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
        self.landslide_reg.fit(X, df["landslide_prob"])
        
        self.flood_reg = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
        self.flood_reg.fit(X, df["flood_prob"])

    def predict_segment_risk(self, rainfall_intensity_mm, rain_accum_24h_mm, slope_deg, elevation_m, distance_to_river_m, historical_landslide_count, road_condition_index):
        # Format the input
        X = pd.DataFrame([[
            rainfall_intensity_mm,
            rain_accum_24h_mm,
            slope_deg,
            elevation_m,
            distance_to_river_m,
            historical_landslide_count,
            road_condition_index
        ]], columns=self.feature_names)
        
        # Predictions
        closure_prob = self.closure_clf.predict_proba(X)[0][1]
        landslide_prob = self.landslide_reg.predict(X)[0]
        flood_prob = self.flood_reg.predict(X)[0]
        
        # Feature Importance (Proxy using standard model importances directly based on local single-prediction trees)
        # For a truly explainable model per-prediction, we identify which features of THIS instance deviate the most.
        # But a lightweight substitute: multiply trained feature importances by the normalized feature values.
        clf_importances = self.closure_clf.feature_importances_
        raw_vals = X.iloc[0].values
        
        # very naive local importance proportional to magnitude and global weight
        norm_vals = raw_vals / (np.max(self._generate_synthetic_data()[self.feature_names].values, axis=0) + 1e-6)
        local_importance = clf_importances * norm_vals
        local_importance = local_importance / (np.sum(local_importance) + 1e-6)
        
        ranked_indices = np.argsort(local_importance)[::-1]
        
        feature_importance = {}
        for i in range(3): # Top 3
            idx = ranked_indices[i]
            feat_name = self.feature_names[idx]
            disp_name = self.feature_display_names[feat_name]
            feature_importance[disp_name] = round(local_importance[idx], 2)
            
        # Confidence score (how pure the terminal nodes were for this prediction)
        probs = self.closure_clf.predict_proba(X)[0]
        confidence_score = float(np.max(probs))
        # Window
        risk_window = "Next 4–8 Hours"
        if closure_prob > 0.8:
            risk_window = "Imminent (0-2 Hours)"
        elif closure_prob < 0.2:
            risk_window = "Clear (Next >12 Hours)"

        return {
            "closure_probability": round(closure_prob, 3),
            "landslide_probability": round(landslide_prob, 3),
            "flood_probability": round(flood_prob, 3),
            "risk_window": risk_window,
            "feature_importance": feature_importance,
            "confidence_score": round(confidence_score, 3)
        }
