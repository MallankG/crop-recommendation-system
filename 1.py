from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import ee
import geemap
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import json
import base64
import lime.lime_tabular
import pandas as pd
from io import BytesIO
import traceback
import numpy as np
from groq import Groq
import joblib
from sklearn.preprocessing import StandardScaler
import time
import concurrent.futures
import uuid
import datetime


app = Flask(__name__)
CORS(app)

# Load models once at startup
ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Earth Engine
ee.Initialize(project='ee-mallankg')

# Cache to store results for regions
region_cache = {}

def plot_to_json(fig):
    """Convert Matplotlib figure to base64 JSON with compression."""
    buf = BytesIO()
    # Use lower DPI for smaller size
    fig.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close figure to free memory
    return img_str

start_date = '2019-01-01'
end_date = '2024-01-01'

# Function to Extract Time-Series Data
def extract_timeseries(point, dataset, band_name, reducer=ee.Reducer.mean()):
    """Extracts the latest time-series data for a given dataset and band."""
    try:
        collection = (
            ee.ImageCollection(dataset)
            .filterBounds(point)
            .filterDate(ee.Date(start_date), ee.Date(end_date))
            .select(band_name)
            .sort('system:time_start', False)  # Sort descending (latest first)
        )
        latest_image = ee.Image(collection.first())  # Get the most recent image
        latest_date = latest_image.get('system:time_start')
        latest_value = latest_image.reduceRegion(reducer=reducer, geometry=point, scale=500).get(band_name)
        # Log the value being fetched
        return ee.Feature(None, {'date': latest_date, 'value': latest_value}).getInfo()
    except Exception as e:
        print(f"Error extracting time series for {dataset}/{band_name}: {e}")
        return {'value': None}


def predict_crop_basic(rainfall_data, temperature_data, soil_moisture_data, ndvi_mean, npkph=None):
    """Basic crop prediction without LIME or Groq - returns quickly"""
    try:
        df = pd.read_csv('2.csv')
        X = df.drop('label', axis=1)
        new_sample = X.iloc[0].copy()
        
        # Set all features
        for feature in X.columns:
            if feature.lower() == 'humidity':
                new_sample[feature] = soil_moisture_data['value'] if soil_moisture_data and 'value' in soil_moisture_data else X[feature].mean()
            elif feature.lower() == 'rainfall':
                new_sample[feature] = rainfall_data['value'] if rainfall_data and 'value' in rainfall_data else X[feature].mean()
            elif feature.lower() == 'temperature':
                new_sample[feature] = temperature_data['value'] if temperature_data and 'value' in temperature_data else X[feature].mean()
            elif feature.lower() == 'ndvi':
                new_sample[feature] = ndvi_mean if ndvi_mean is not None else X[feature].mean()
            elif npkph and feature in ['N', 'P', 'K', 'ph']:
                key = feature if feature != 'ph' else 'pH'
                val = npkph.get(key) or npkph.get(key.upper()) or npkph.get(key.lower())
                if val is not None:
                    new_sample[feature] = val
                else:
                    new_sample[feature] = X[feature].mean()
        
        new_sample_scaled = scaler.transform([new_sample])
        prediction_proba = ensemble_model.predict_proba(new_sample_scaled)
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        top_3_predictions = ensemble_model.classes_[top_3_indices]
        top_3_probas = prediction_proba[0][top_3_indices]
        
        # Return minimum information for quick response
        response = {
            "predictions": [{
                "class": pred,
                "probability": float(prob)
            } for pred, prob in zip(top_3_predictions, top_3_probas)],
            "explanations": "Detailed explanations will be available shortly...",
            "ai_interpretation": "AI interpretation loading...",
            "feature_importances": []
        }
        
        return response, new_sample, new_sample_scaled, top_3_indices, X
    except Exception as e:
        print(f"Error in basic crop prediction: {str(e)}")
        return {
            "predictions": [{"class": "Unknown", "probability": 0}],
            "explanations": f"Error in prediction: {str(e)}",
            "ai_interpretation": "Unable to provide interpretation due to processing error."
        }, None, None, None, None


def generate_lime_explanations(new_sample_scaled, new_sample, X, top_3_indices):
    """Generate LIME explanations for the top 3 predictions"""
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=scaler.transform(X),
            feature_names=X.columns,
            class_names=ensemble_model.classes_,
            mode='classification'
        )
        explanations = []
        feature_importances = []
        for i in range(min(3, len(top_3_indices))):
            if not isinstance(top_3_indices[i], (int, np.integer)) or top_3_indices[i] < 0 or top_3_indices[i] >= len(ensemble_model.classes_):
                print(f"Invalid class index for LIME explanation: {top_3_indices[i]}, max index: {len(ensemble_model.classes_)-1}")
                continue
            try:
                exp = explainer.explain_instance(
                    new_sample_scaled[0],
                    ensemble_model.predict_proba,
                    num_features=8,  # Show more features
                    top_labels=1,
                    labels=[int(top_3_indices[i])]
                )
                explanation = exp.as_list(label=int(top_3_indices[i]))
                # Always include P and K in the explanation
                features_in_exp = dict(explanation)
                for f in ['P', 'K']:
                    if f not in features_in_exp:
                        features_in_exp[f] = 0.0
                sorted_exp = sorted(features_in_exp.items(), key=lambda x: abs(x[1]), reverse=True)
                explanations.append({
                    "prediction": ensemble_model.classes_[top_3_indices[i]],
                    "explanation": sorted_exp
                })
                feature_importances.append({
                    "crop": ensemble_model.classes_[top_3_indices[i]],
                    "features": [
                        {"feature": feature, "importance": float(weight)} for feature, weight in sorted_exp
                    ]
                })
            except Exception as e:
                print(f"Error generating explanation for index {top_3_indices[i]}: {str(e)}")
                continue
        if not explanations:
            raise ValueError("Could not generate any valid explanations")
        formatted_explanations = ""
        for i, exp in enumerate(explanations):
            formatted_explanations += f"\nExplanation for {exp['prediction']}:\n"
            for feature, weight in exp['explanation']:
                formatted_explanations += f"- {feature}: {weight:.4f}\n"
        return formatted_explanations, feature_importances
    except Exception as e:
        print(f"Error generating LIME explanations: {str(e)}")
        return f"Error generating explanations: {str(e)}", []


def generate_groq_interpretation(formatted_explanations, top_predictions):
    """Generate AI interpretation using Groq API"""
    try:
        # Format the predictions for Groq
        predictions_text = "\n".join([
            f"Crop: {pred['class']}, Probability: {pred['probability']:.4f}"
            for pred in top_predictions
        ])
        
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {'role': 'system', 'content': 'You are an expert in agricultural ML model interpretation. Provide concise conclusions about crop suitability based on the predictions and feature importance.'},
                {'role': 'user', 'content': f"Analyze these crop predictions and feature explanations:\n\nPredictions:\n{predictions_text}\n\nFeature Explanations:\n{formatted_explanations}\n\nProvide a concise interpretation (3-4 sentences) explaining what these results mean for a farmer."}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=1,
            stream=False
        )
        
        interpretation = completion.choices[0].message.content
        return interpretation
    except Exception as e:
        print(f"Error generating Groq interpretation: {str(e)}")
        return f"AI interpretation unavailable: {str(e)}"


@app.route('/predict', methods=['POST'])
def predict_yield():
    start_time = time.time()
    request_id = f"request_{start_time}"
    print(f"[{request_id}] Starting request processing at {datetime.datetime.now().isoformat()}")
    try:
        data = request.json
        roi_coordinates = data.get("roi")
        if not roi_coordinates:
            return jsonify({"error": "Region of Interest (ROI) not provided"}), 400

        # Generate a cache key based on coordinates
        cache_key = str(sorted(map(tuple, roi_coordinates)))
        # Check cache first
        if cache_key in region_cache:
            print(f"[{request_id}] Cache hit! Returning cached result")
            return jsonify(region_cache[cache_key])

        latitude = np.mean([coord[1] for coord in roi_coordinates])
        longitude = np.mean([coord[0] for coord in roi_coordinates])
        point = ee.Geometry.Point([longitude, latitude])
        roi = ee.Geometry.Polygon(roi_coordinates)

        # Apply cloud mask to Sentinel-2 images (using a smaller date range to speed up)
        def maskS2clouds(image):
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)
        
        # Use a smaller date range and more aggressive cloud filtering
        three_years_ago = time.strftime('%d-%m-%y', time.localtime(time.time() - 1095*24*60*60))
        print(f"[{request_id}] Using date range: {three_years_ago} to now")
        
        # Use cloud-masked collection with smaller date range
        s2_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate(three_years_ago, time.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5)) \
            .map(maskS2clouds)
        
        dataset = s2_collection.median()

        if dataset is None:
            return jsonify({"error": "No satellite data found"}), 404

        # Compute NDVI
        ndvi = dataset.normalizedDifference(["B8", "B4"]).rename("NDVI")

        # Compute mean NDVI over ROI
        ndvi_stats = (
            ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=30, maxPixels=1e9
            ).getInfo()
        )

        if not ndvi_stats or "NDVI" not in ndvi_stats:
            return jsonify({"error": "Failed to compute NDVI"}), 500

        ndvi_mean = ndvi_stats["NDVI"]
        print(f"[{request_id}] NDVI mean: {ndvi_mean}")

        # Fetch Weather Data (Temperature, Rainfall, Soil Moisture) and SoilGrids N, P, K, pH in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            future_rainfall = executor.submit(extract_timeseries, point, 'UCSB-CHG/CHIRPS/DAILY', 'precipitation')
            future_temperature = executor.submit(extract_timeseries, point, 'MODIS/061/MOD11A2', 'LST_Day_1km')
            future_soil = executor.submit(extract_timeseries, point, 'NASA/SMAP/SPL4SMGP/007', 'sm_surface')
            future_n = executor.submit(lambda: ee.Image('projects/soilgrids-isric/nitrogen_mean').select('nitrogen_0-5cm_mean').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8).getInfo().get('nitrogen_0-5cm_mean'))
            future_p = executor.submit(lambda: ee.Image('projects/soilgrids-isric/phosphorus_mean').select('phosphorus_0-5cm_mean').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8).getInfo().get('phosphorus_0-5cm_mean'))
            future_k = executor.submit(lambda: ee.Image('projects/soilgrids-isric/potassium_mean').select('potassium_0-5cm_mean').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8).getInfo().get('potassium_0-5cm_mean'))
            future_ph = executor.submit(lambda: ee.Image('projects/soilgrids-isric/phh2o_mean').select('phh2o_0-5cm_mean').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8).getInfo().get('phh2o_0-5cm_mean'))
            try:
                rainfall_data = future_rainfall.result()
            except Exception:
                rainfall_data = {'value': None}
            try:
                temperature_data = future_temperature.result()
            except Exception:
                temperature_data = {'value': None}
            try:
                soil_moisture_data = future_soil.result()
            except Exception:
                soil_moisture_data = {'value': None}
            npkph = {}
            try:
                npkph['N'] = future_n.result()
            except Exception:
                npkph['N'] = None
            try:
                npkph['P'] = future_p.result()
            except Exception:
                npkph['P'] = None
            try:
                npkph['K'] = future_k.result()
            except Exception:
                npkph['K'] = None
            try:
                npkph['pH'] = future_ph.result()
            except Exception:
                npkph['pH'] = None
        print('SoilGrids NPKpH:', npkph)

        # Fallback: If P or K is None, use mean from 2.csv
        if npkph.get('P') is None or npkph.get('K') is None:
            try:
                df_fallback = pd.read_csv('2.csv')
                if npkph.get('P') is None and 'P' in df_fallback.columns:
                    npkph['P'] = df_fallback['P'].mean()
                    print(f"[Fallback] Using mean P from 2.csv: {npkph['P']}")
                if npkph.get('K') is None and 'K' in df_fallback.columns:
                    npkph['K'] = df_fallback['K'].mean()
                    print(f"[Fallback] Using mean K from 2.csv: {npkph['K']}")
            except Exception as e:
                print(f"[Fallback] Error loading mean P/K from 2.csv: {e}")

        # Log N, P, K, pH values
        print(f"[{request_id}] NPKpH values: N={npkph.get('N')}, P={npkph.get('P')}, K={npkph.get('K')}, pH={npkph.get('pH')}")
        # Log current time
        print(f"[{request_id}] Current time: {datetime.datetime.now().isoformat()}")

        # Define yield prediction image with simpler visualization
        yield_image = ndvi.multiply(20).rename("Yield")

        # Visualization parameters
        yield_vis_params = {
            "min": 0.3,
            "max": 0.7,
            "palette": ["#ff0000", "#ffd758", "#37ab3e", "#53ff30", "#185a1f"]
        }

        # Use a higher scale (less detail) for faster processing
        scale = 30
        
        # Run LIME and Groq synchronously
        basic_prediction_result, new_sample, new_sample_scaled, top_3_indices, X = predict_crop_basic(
            rainfall_data, temperature_data, soil_moisture_data, ndvi_mean, npkph=npkph
        )
        formatted_explanations, feature_importances = generate_lime_explanations(new_sample_scaled, new_sample, X, top_3_indices)
        ai_interpretation = generate_groq_interpretation(formatted_explanations, basic_prediction_result['predictions'])
        
        # Attach LIME and Groq results
        basic_prediction_result['explanations'] = formatted_explanations
        basic_prediction_result['ai_interpretation'] = ai_interpretation
        basic_prediction_result['feature_importances'] = feature_importances
        predicted_yield = basic_prediction_result['predictions'][0]['probability'] if basic_prediction_result['predictions'] else None
        try:
            # Simplified image fetching with fixed scale
            base_map = geemap.ee_to_numpy(
                dataset.select(["B4", "B3", "B2"]), 
                region=roi, 
                scale=scale
            )
            
            yield_map = geemap.ee_to_numpy(
                yield_image.visualize(**yield_vis_params), 
                region=roi, 
                scale=scale
            )
            
            # Stretch base_map for visualization if values are too low
            base_map_min = np.min(base_map)
            base_map_max = np.max(base_map)
            if base_map_max - base_map_min > 0:
                base_map = (base_map - base_map_min) / (base_map_max - base_map_min)
            else:
                base_map = np.zeros_like(base_map)

            # Create visualization - smaller figure size, lower quality
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(base_map)
            axs[0].set_title("Satellite Image")
            axs[0].axis("off")

            axs[1].imshow(yield_map)
            axs[1].set_title("Predicted Yield")
            axs[1].axis("off")

            # Convert the figure to JSON (with compression)
            plot_json = plot_to_json(fig)
            
        except Exception as e:
            print(f"[{request_id}] Error fetching maps: {e}")
            plot_json = ""
        # Use actual values for temperature, rainfall, and soil moisture
        temp_val = temperature_data.get('value') if temperature_data and 'value' in temperature_data else None
        rain_val = rainfall_data.get('value') if rainfall_data and 'value' in rainfall_data else None
        soil_val = soil_moisture_data.get('value') if soil_moisture_data and 'value' in soil_moisture_data else None
        # Create result
        result = {
            "ndvi_mean": ndvi_mean,
            "predicted_yield": predicted_yield,
            "temperature_C": temp_val,
            "rainfall_mm": rain_val,
            "soil_moisture": soil_val,
            "prediction": basic_prediction_result,
            "image": plot_json
        }
        # Log result without the image and AI interpretation
        log_result = {k: v for k, v in result.items() if k not in ('image', 'prediction')}
        # Log prediction without ai_interpretation
        if 'prediction' in result and isinstance(result['prediction'], dict):
            log_result['prediction'] = {k: v for k, v in result['prediction'].items() if k != 'ai_interpretation'}
        elapsed_time = time.time() - start_time
        print(f"[{request_id}] Request processed in {elapsed_time:.2f} seconds")
        return jsonify(result)

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[{request_id}] Error after {elapsed_time:.2f} seconds:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "partial_results": 'ndvi_mean' in locals(),
            "ndvi_mean": ndvi_mean if 'ndvi_mean' in locals() else None
        }), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
