export interface PredictionResult {
  ndvi_mean: number
  predicted_yield: number | null
  temperature_C: number
  rainfall_mm: number
  soil_moisture: number
  image: string
  prediction: {
    predictions: Array<{
      class: string
      probability: number
    }>
    explanations: string
    ai_interpretation: string
    feature_importances?: Array<{
      crop: string
      features: Array<{
        feature: string
        importance: number
      }>
    }>
  }
  selectedCrop?: string
  task_id?: string
  mean_evi?: number;
}

export interface BatchPredictionResult {
  id: string
  name: string
  status: "pending" | "processing" | "completed" | "failed"
  result?: PredictionResult
  error?: string
  coordinates: number[][]
}

export interface CropOption {
  value: string
  label: string
}
