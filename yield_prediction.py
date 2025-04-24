from flask import Flask, request, jsonify
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
import requests
import datetime
import rasterio
import json
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

# Load sensitive data from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
EARTH_ENGINE_PROJECT = os.getenv('EARTH_ENGINE_PROJECT')

ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

# Authenticate and initialize Earth Engine
ee.Initialize(project=EARTH_ENGINE_PROJECT)

def plot_to_json(fig):
    """Convert Matplotlib figure to base64 JSON."""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

start_date = '2019-01-01'
end_date = '2024-01-01'

# Function to Extract Time-Series Data
def extract_timeseries(point, dataset, band_name, reducer=ee.Reducer.mean()):
    """Extracts the latest time-series data for a given dataset and band."""
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
    
    return ee.Feature(None, {'date': latest_date, 'value': latest_value}).getInfo()


def predict_crop(rainfall_data, temperature_data, soil_moisture_data, ndvi_mean, npkph=None):
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
                print(f"[SoilGrids] {feature} fetched from dataset: {val}")
                new_sample[feature] = val
            else:
                mean_val = X[feature].mean()
                print(f"[SoilGrids] {feature} missing, using mean: {mean_val}")
                new_sample[feature] = mean_val
    # Fill missing values with column means
    for feature in X.columns:
        if pd.isnull(new_sample[feature]) or new_sample[feature] is None:
            new_sample[feature] = X[feature].mean()
    print('Model input sample before scaling:', new_sample)
    new_sample_df = pd.DataFrame([new_sample], columns=X.columns)
    new_sample_scaled = scaler.transform(new_sample_df)
    prediction_proba = ensemble_model.predict_proba(new_sample_scaled)
    top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
    top_3_predictions = ensemble_model.classes_[top_3_indices]
    top_3_probas = prediction_proba[0][top_3_indices]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=scaler.transform(X),
        feature_names=X.columns,
        class_names=ensemble_model.classes_,
        mode='classification'
    )
    explanations = []
    for i in range(3):
        exp = explainer.explain_instance(new_sample_scaled[0], ensemble_model.predict_proba, num_features=5, top_labels=3)
        explanation = exp.as_list(label=top_3_indices[i])
        explanations.append({
            "prediction": top_3_predictions[i],
            "probability": top_3_probas[i],
            "explanation": explanation
        })
    formatted_explanations = ""
    for i, exp in enumerate(explanations):
        formatted_explanations += f"\nExplanation for top {i+1} prediction ({exp['prediction']} with probability {exp['probability']:.4f}):\n"
        for feature, weight in exp['explanation']:
            formatted_explanations += f"- {feature}: {weight:.4f}\n"
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {'role': 'system', 'content': 'You are an expert in ML model interpretation. Provide concise conclusions.'},
            {'role': 'system', 'content': formatted_explanations}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    response = {
        "predictions": [{
            "class": pred,
            "probability": float(prob)
        } for pred, prob in zip(top_3_predictions, top_3_probas)],
        "explanations": formatted_explanations,
        "ai_interpretation": completion.choices[0].message.content
    }
    return (response)

def fetch_openweather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        temperature = data['main']['temp']
        rainfall = data.get('rain', {}).get('1h', 0)  # mm in last 1h
        # OpenWeather does not provide soil moisture directly; set to None or use another API if needed
        soil_moisture = None
        return {
            'temperature': temperature,
            'rainfall': rainfall,
            'soil_moisture': soil_moisture
        }
    return {'temperature': None, 'rainfall': None, 'soil_moisture': None}

def reverse_geocode(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=10&addressdetails=1"
        resp = requests.get(url, headers={"User-Agent": "crop-yield-prediction/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            # Try to get district, state, or city
            address = data.get('address', {})
            for key in ['city', 'town', 'village', 'county', 'state_district', 'state']:
                if key in address:
                    return address[key]
        return None
    except Exception as e:
        print('Reverse geocoding error:', e)
        return None

def extract_mean_from_tiff(tiff_path, roi_coordinates):
    # Convert ROI to GeoJSON polygon
    roi_geojson = {
        "type": "Polygon",
        "coordinates": [roi_coordinates]
    }
    with rasterio.open(tiff_path) as src:
        # Rasterio expects coordinates in (lon, lat)
        out_image, out_transform = rasterio.mask.mask(src, [roi_geojson], crop=True)
        data = out_image[0]
        data = data[data != src.nodata]
        return float(np.nanmean(data)) if data.size > 0 else None

@app.route('/predict', methods=['POST'])
def predict_yield():
    try:
        data = request.json
        print("Received Data:", data)  # Debugging

        roi_coordinates = data.get("roi")
        if not roi_coordinates:
            return jsonify({"error": "Region of Interest (ROI) not provided"}), 400

        print("ROI Coordinates:", roi_coordinates)
        latitude = np.mean([coord[1] for coord in roi_coordinates])
        longitude = np.mean([coord[0] for coord in roi_coordinates])
        # Detect region name
        # region_name = reverse_geocode(latitude, longitude)
        # print('Detected region:', region_name)
        # Get current month/season
        # now = datetime.datetime.now()
        # month = now.strftime('%B')
        # Query Groq for region-crop-season info
        # CROP_LIST = [
        #     'cotton', 'apple', 'mungbean', 'pomegranate', 'rice', 'mango', 'muskmelon',
        #     'pigeonpeas', 'orange', 'grapes', 'watermelon', 'banana', 'maize', 'blackgram',
        #     'coffee', 'chickpea', 'kidneybeans', 'papaya', 'mothbeans', 'coconut', 'jute', 'lentil'
        # ]
        # region_crop = None
        # if region_name:
        #     groq_prompt = (
        #         f"Given this list of crops: {', '.join(CROP_LIST)}. "
        #         f"What is the single most predominant crop grown in {region_name}, India during {month}? "
        #         f"Only choose from the list above. Return only the crop name, no explanation."
        #     )
        #     client = Groq(api_key=GROQ_API_KEY)
        #     groq_resp = client.chat.completions.create(
        #         model="llama3-8b-8192",
        #         messages=[
        #             {'role': 'system', 'content': 'You are an expert in Indian agriculture.'},
        #             {'role': 'user', 'content': groq_prompt}
        #         ],
        #         temperature=0.2,
        #         max_tokens=16,
        #         top_p=1,
        #         stream=False
        #     )
        #     raw_resp = groq_resp.choices[0].message.content.strip().lower()
        #     print('Groq raw response:', raw_resp)
        #     region_crop = next((crop for crop in CROP_LIST if crop == raw_resp), None)
        #     print('Groq region crop:', region_crop)
        #     if region_crop not in CROP_LIST:
        #         region_crop = None

        # Fetch environmental data from OpenWeather
        openweather = fetch_openweather_data(latitude, longitude)
        temperature_data = {'value': openweather['temperature']}
        rainfall_data = {'value': openweather['rainfall']}
        soil_moisture_data = {'value': openweather['soil_moisture']}

        # Define the ROI
        roi = ee.Geometry.Polygon(roi_coordinates)

        # Apply cloud mask to Sentinel-2 images
        def maskS2clouds(image):
            qa = image.select('QA60')
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)
        
        # Use cloud-masked collection
        s2_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate("2024-01-01", "2024-12-31") \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)) \
            .map(maskS2clouds)
        n_images = s2_collection.size().getInfo()
        print("Number of Sentinel-2 images after filtering:", n_images)
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
        print("NDVI stats:", ndvi_stats)

        if not ndvi_stats or "NDVI" not in ndvi_stats:
            return jsonify({"error": "Failed to compute NDVI"}), 500

        ndvi_mean = ndvi_stats["NDVI"]
        print("NDVI mean:", ndvi_mean)

        # Fetch SoilGrids N, P, K, pH for the ROI (using correct asset paths and band names, fetch individually)
        npkph = {}
        # Nitrogen
        try:
            nitrogen_img = ee.Image('projects/soilgrids-isric/nitrogen_mean')
            n_band = 'nitrogen_0-5cm_mean'
            n_mean = nitrogen_img.select(n_band).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8
            ).getInfo().get(n_band)
            npkph['N'] = n_mean
        except Exception as e:
            print('SoilGrids N fetch error:', e)
            npkph['N'] = None
        # Potassium
        try:
            africa_lat_min, africa_lat_max = -35, 37
            africa_lon_min, africa_lon_max = -18, 52
            if africa_lat_min <= latitude <= africa_lat_max and africa_lon_min <= longitude <= africa_lon_max:
                potassium_img = ee.Image('ISDASOIL/Africa/v1/potassium_extractable').select('mean_0_20')
                k_mean = potassium_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8
                ).get('mean_0_20')
                if k_mean is not None:
                    k_mean = ee.Number(k_mean).getInfo()
                npkph['K'] = k_mean
            else:
                df = pd.read_csv('2.csv')
                if 'K' in df.columns:
                    npkph['K'] = df['K'].mean()
                else:
                    npkph['K'] = None
        except Exception as e:
            print('Potassium fetch error:', e)
            npkph['K'] = None
        # Phosphorus
        try:
            if africa_lat_min <= latitude <= africa_lat_max and africa_lon_min <= longitude <= africa_lon_max:
                phosphorus_img = ee.Image('ISDASOIL/Africa/v1/phosphorus_extractable').select('mean_0_20')
                p_mean = phosphorus_img.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8
                ).get('mean_0_20')
                if p_mean is not None:
                    p_mean = ee.Number(p_mean).getInfo()
                npkph['P'] = p_mean
            else:
                df = pd.read_csv('2.csv')
                if 'P' in df.columns:
                    npkph['P'] = df['P'].mean()
                else:
                    npkph['P'] = None
        except Exception as e:
            print('Phosphorus fetch error:', e)
            npkph['P'] = None
        # pH
        try:
            phh2o_img = ee.Image('projects/soilgrids-isric/phh2o_mean')
            ph_band = 'phh2o_0-5cm_mean'
            ph_mean = phh2o_img.select(ph_band).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=roi, scale=250, maxPixels=1e8
            ).getInfo().get(ph_band)
            npkph['pH'] = ph_mean
        except Exception as e:
            print('SoilGrids pH fetch error:', e)
            npkph['pH'] = None
        print('SoilGrids NPKpH:', npkph)

        # Define yield prediction image
        yield_image = ndvi.multiply(20).rename("Yield")

        # Visualization parameters
        yield_vis_params = {
            "min": 0.3,
            "max": 0.7,
            "palette": ["#ff0000", "#ffd758", "#37ab3e", "#53ff30", "#185a1f"]
        }

        # Try to fetch base_map and yield_map with adaptive scale
        def fetch_numpy_with_adaptive_scale(image, region, bands=None, vis_params=None, initial_scale=10, max_scale=100, step=10):
            scale = initial_scale
            while scale <= max_scale:
                try:
                    if bands:
                        arr = geemap.ee_to_numpy(image.select(bands), region=region, scale=scale)
                        return arr, scale
                    elif vis_params:
                        arr = geemap.ee_to_numpy(image.visualize(**vis_params), region=region, scale=scale)
                        return arr, scale
                except Exception as e:
                    if 'must be less than or equal to' in str(e) and scale < max_scale:
                        print(f"Request too large at scale={scale}, trying scale={scale+step}")
                        scale += step
                    else:
                        print(f"Failed to fetch image at scale={scale}: {e}")
                        return None, scale
            return None, scale

        # Print B8 and B4 band values for NDVI debugging
        try:
            b8_img = dataset.select(["B8"])
            b4_img = dataset.select(["B4"])
            b8_arr, _ = fetch_numpy_with_adaptive_scale(b8_img, roi, bands=["B8"], initial_scale=30)
            b4_arr, _ = fetch_numpy_with_adaptive_scale(b4_img, roi, bands=["B4"], initial_scale=30)
            if b8_arr is not None and b4_arr is not None:
                print("B8 min/max:", np.min(b8_arr), np.max(b8_arr))
                print("B4 min/max:", np.min(b4_arr), np.max(b4_arr))
                print("B8 sample values:", b8_arr.flatten()[:10])
                print("B4 sample values:", b4_arr.flatten()[:10])
        except Exception as e:
            print("Error fetching B8/B4 for NDVI debug:", e)

        # Use adaptive scale for base_map and yield_map
        base_map, used_scale_base = fetch_numpy_with_adaptive_scale(dataset, roi, bands=["B4", "B3", "B2"], initial_scale=10, max_scale=100, step=10)
        yield_map, used_scale_yield = fetch_numpy_with_adaptive_scale(yield_image, roi, vis_params=yield_vis_params, initial_scale=10, max_scale=100, step=10)
        print(f"Used scale for base_map: {used_scale_base}, yield_map: {used_scale_yield}")

        if base_map is None or yield_map is None:
            return jsonify({"error": "Failed to generate image visuals. Try a smaller region or coar"})

        # Stretch base_map for visualization if values are too low
        base_map_min = np.min(base_map)
        base_map_max = np.max(base_map)
        if base_map_max - base_map_min > 0:
            base_map = (base_map - base_map_min) / (base_map_max - base_map_min)
        else:
            base_map = np.zeros_like(base_map)
        print("Base map after stretching min/max:", np.min(base_map), np.max(base_map))

        # Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(12, 12))
        axs[0].imshow(base_map)
        axs[0].set_title("Satellite Image (True Color)")
        axs[0].axis("off")

        axs[1].imshow(yield_map)
        axs[1].set_title("Predicted Crop Yield")
        axs[1].axis("off")

        # Convert the figure to JSON
        plot_json = plot_to_json(fig)
        prediction = predict_crop(rainfall_data, temperature_data, soil_moisture_data, ndvi_mean, npkph=npkph)
        # Save the region crop and reorder predictions
        # region_specific_crop = region_crop
        # if region_specific_crop and 'predictions' in prediction:
        #     pred_list = prediction['predictions']
        #     region_pred = next((p for p in pred_list if p['class'].lower().strip() == region_specific_crop.strip()), None)
        #     pred_list = [p for p in pred_list if p['class'].lower().strip() != region_specific_crop.strip()]
        #     next_highest_prob = max([p['probability'] for p in pred_list], default=0)
        #     if region_pred:
        #         region_prob = max(region_pred['probability'], min(next_highest_prob + 0.01, 1.0))
        #     else:
        #         region_prob = min(next_highest_prob + 0.01, 1.0)
        #     pred_list.insert(0, {'class': region_specific_crop, 'probability': region_prob})
        #     prediction['predictions'] = pred_list[:3]
        #     print('Predictions after region crop prioritization:', prediction['predictions'])

        # Use the ML model's top predicted crop probability as the yield (or adjust as needed)
        predicted_yield = prediction['predictions'][0]['probability'] if 'predictions' in prediction and prediction['predictions'] else None

        # Extract LIME feature importance for the top prediction
        lime_feature_importance = None
        feature_importances = []
        if 'explanations' in prediction and isinstance(prediction['explanations'], str):
            import re
            # Adjust regex to be more robust to whitespace and case
            match = re.search(r"Explanation for top 1 prediction.*?:\s*((?:- .+?: .+?\n)+)", prediction['explanations'], re.IGNORECASE)
            if match:
                lines = match.group(1).strip().split('\n')
                lime_feature_importance = [
                    {'feature': l.split(':')[0].replace('- ', '').strip(), 'weight': float(l.split(':')[1].strip())}
                    for l in lines if ':' in l
                ]
                feature_importances = [{
                    'crop': prediction['predictions'][0]['class'] if 'predictions' in prediction and prediction['predictions'] else 'Top Prediction',
                    'features': [
                        {'feature': f['feature'], 'importance': f['weight']} for f in lime_feature_importance
                    ]
                }]

        # Environmental data values
        temperature_val = temperature_data['value']
        rainfall_val = rainfall_data['value']
        # Fetch soil moisture from NASA SMAP if OpenWeather is None
        if soil_moisture_data['value'] is None:
            try:
                point = ee.Geometry.Point([longitude, latitude])
                sm_data = extract_timeseries(point, "NASA/SMAP/SPL4SMGP/007", "sm_surface")
                soil_moisture_val = sm_data['value'] if sm_data and 'value' in sm_data else None
            except Exception as e:
                print('Soil moisture fallback fetch error:', e)
                soil_moisture_val = None
        else:
            soil_moisture_val = soil_moisture_data['value']

        # Extract mean EVI for ROI before using it in the response
        evi_tiff_path = 'evi_mod13q1.tmwm.inpaint_p.90_250m_s_20201101_20201231_go_epsg.4326_v20230608.tif'
        try:
            mean_evi = extract_mean_from_tiff(evi_tiff_path, roi_coordinates)
        except Exception as e:
            print('Error extracting mean EVI:', e)
            mean_evi = None
        print('Mean EVI:', mean_evi)

        return jsonify({
            "ndvi_mean": ndvi_mean,
            "predicted_yield": predicted_yield,
            "temperature_C": temperature_val,
            "rainfall_mm": rainfall_val,
            "soil_moisture": soil_moisture_val,
            "image": plot_json,
            "prediction": {
                **prediction,
                "feature_importances": feature_importances
            },
            "lime_feature_importance": lime_feature_importance,
            "environmental_data": {
                "temperature_C": temperature_val,
                "rainfall_mm": rainfall_val,
                "soil_moisture": soil_moisture_val
            },
            "soil_moisture": soil_moisture_val,
            "mean_evi": mean_evi,
        })

    except Exception as e:
        print("Error occurred:", str(e))
        print(traceback.format_exc())  # Print full error traceback
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
