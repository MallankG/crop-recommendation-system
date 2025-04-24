# Crop Yield Prediction

This project predicts crop yield using satellite imagery, weather data, and soil properties. It features a backend powered by Python (Flask, Earth Engine, ML models) and a modern frontend (Next.js, React, Tailwind CSS).

## Features
- Predicts crop yield for a selected region using satellite and weather data
- Visualizes satellite images and predicted yield
- Explains predictions with LIME and AI-generated interpretations
- Interactive dashboard for region selection and analysis

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- pnpm (or npm/yarn)
- Google Earth Engine account

### Backend Setup
1. Clone the repository.
2. Create a `.env` file in the root directory with the following:
   ```
   GROQ_API_KEY=your_groq_api_key
   OPENWEATHER_API_KEY=your_openweather_api_key
   EARTH_ENGINE_PROJECT=your_earth_engine_project_id
   ```
3. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the backend:
   ```sh
   python yield_prediction.py
   ```

### Frontend Setup
1. Install dependencies:
   ```sh
   pnpm install
   # or
   npm install
   ```
2. Run the frontend:
   ```sh
   pnpm dev
   # or
   npm run dev
   ```

## Project Structure
- `yield_prediction.py` — Backend API for predictions
- `components/` — React UI components
- `app/` — Next.js app directory
- `public/` — Static assets
- `ensemble_model.pkl`, `scaler.pkl` — ML model files (not tracked in git)

## Notes
- Sensitive keys are stored in `.env` (not tracked by git).
- Model/data files are ignored by `.gitignore`.
- For Earth Engine, ensure you have access and proper credentials.

## License
MIT
