"use client"

import { useState } from "react"
import type { PredictionResult } from "@/lib/types"
import { Download, RefreshCw, Clock } from "lucide-react"
import { jsPDF } from "jspdf"
import InfoTooltip from "./info-tooltip"

interface AnalysisPanelProps {
  selectedRegion: number[][]
  predictionResult: PredictionResult | null
  onSubmit: (npkParams?: { n?: number|null, p?: number|null, k?: number|null, npkMode?: 'auto'|'manual' }) => void
  onReset: () => void
  error: string | null
  savedRegions: { id: string; coordinates: number[][] }[]
  onLoadRegion: (coordinates: number[][]) => void
  backgroundTaskStatus?: 'processing' | 'lime_complete' | 'complete' | 'error' | null
}

export default function AnalysisPanel({
  selectedRegion,
  predictionResult,
  onSubmit,
  onReset,
  error,
  savedRegions,
  onLoadRegion,
  backgroundTaskStatus
}: Omit<AnalysisPanelProps, 'selectedCrop'>) {
  const [activeTab, setActiveTab] = useState("imagery")
  const [showHistory, setShowHistory] = useState(false)
  const [npkMode, setNpkMode] = useState<'auto' | 'manual'>('auto')
  const [manualN, setManualN] = useState<string>("")
  const [manualP, setManualP] = useState<string>("")
  const [manualK, setManualK] = useState<string>("")

  const handleSubmitInternal = () => {
    if (npkMode === 'manual') {
      onSubmit({
        n: manualN ? parseFloat(manualN) : null,
        p: manualP ? parseFloat(manualP) : null,
        k: manualK ? parseFloat(manualK) : null,
        npkMode: 'manual',
      })
    } else {
      onSubmit({ npkMode: 'auto' })
    }
  }

  const handleExportPDF = () => {
    if (!predictionResult) return

    const doc = new jsPDF()

    // Add title
    doc.setFontSize(20)
    doc.text("Crop Yield Prediction Report", 20, 20)

    // Add date
    doc.setFontSize(12)
    doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 20, 30)

    // Add region coordinates
    doc.setFontSize(14)
    doc.text("Region of Interest:", 20, 40)
    doc.setFontSize(10)
    doc.text(
      `Center: Lat ${selectedRegion[0][1].toFixed(4)}, Lng ${selectedRegion[0][0].toFixed(4)}`,
      20,
      50,
    )

    // Add NDVI data
    doc.setFontSize(14)
    doc.text("NDVI Data:", 20, 60)
    doc.setFontSize(10)
    doc.text(`Mean NDVI: ${predictionResult.ndvi_mean.toFixed(4)}`, 20, 70)

    // Add prediction results
    doc.setFontSize(14)
    doc.text("Crop Predictions:", 20, 80)
    doc.setFontSize(10)

    predictionResult.prediction.predictions.forEach((pred, index) => {
      doc.text(
        `${index + 1}. ${pred.class} (${(pred.probability * 100).toFixed(2)}%)`,
        20,
        90 + index * 10,
      )
    })

    // Add environmental data
    doc.setFontSize(14)
    doc.text("Environmental Data:", 20, 100)
    doc.setFontSize(10)
    doc.text(`Temperature: ${predictionResult.temperature_C ?? 'N/A'} °C`, 20, 110)
    doc.text(`Rainfall: ${predictionResult.rainfall_mm ?? 'N/A'} mm`, 20, 120)
    doc.text(`Soil Moisture: ${predictionResult.soil_moisture ?? 'N/A'} %`, 20, 130)

    // Add predicted yield
    doc.setFontSize(14)
    doc.text("Predicted Yield:", 20, 140)
    doc.setFontSize(10)
    doc.text(`${predictionResult.predicted_yield !== null && predictionResult.predicted_yield !== undefined ? (predictionResult.predicted_yield * 100).toFixed(2) + '%' : 'N/A'}`, 20, 150)

    // Add feature importance
    if (predictionResult.prediction.feature_importances && predictionResult.prediction.feature_importances.length > 0) {
      doc.setFontSize(14)
      doc.text("Feature Importance (LIME):", 20, 160)
      doc.setFontSize(10)
      let y = 170
      predictionResult.prediction.feature_importances.forEach((crop) => {
        doc.text(`Crop: ${crop.crop}`, 20, y)
        y += 8
        crop.features.forEach((feature) => {
          doc.text(`- ${feature.feature}: ${feature.importance.toFixed(4)}`, 25, y)
          y += 8
        })
        y += 4
      })
    }

    // Add explanations
    doc.setFontSize(14)
    doc.text("Model Explanations:", 20, 200)
    doc.setFontSize(10)
    doc.text(predictionResult.prediction.explanations || 'N/A', 20, 210, { maxWidth: 170 })

    // Add AI interpretation
    doc.setFontSize(14)
    doc.text("AI Interpretation:", 20, 250)
    doc.setFontSize(10)
    doc.text(predictionResult.prediction.ai_interpretation || 'N/A', 20, 260, { maxWidth: 170 })

    // Add image if available
    if (predictionResult.image) {
      doc.addPage()
      doc.text("Satellite Imagery and Yield Prediction", 20, 20)
      doc.addImage(`data:image/png;base64,${predictionResult.image}`, "PNG", 20, 30, 170, 170)
    }

    // Save the PDF
    doc.save("crop-yield-prediction.pdf")
  }

  return (
    <div className="h-full flex flex-col text-gray-900">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-gray-900 mb-2">Analysis Panel</h2>
        {/* NPK Input Section */}
        {!predictionResult && (
          <div className="mb-4">
            <h3 className="font-medium mb-2">NPK Data Input</h3>
            <div className="flex space-x-4 mb-2">
              <button
                className={`px-3 py-1 rounded ${npkMode === 'auto' ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                onClick={() => setNpkMode('auto')}
              >
                Automatic
              </button>
              <button
                className={`px-3 py-1 rounded ${npkMode === 'manual' ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                onClick={() => setNpkMode('manual')}
              >
                Manual
              </button>
            </div>
            {npkMode === 'manual' && (
              <div className="flex space-x-2">
                <div>
                  <label className="block text-xs text-gray-600">N (mg/kg)</label>
                  <input type="number" value={manualN} onChange={e => setManualN(e.target.value)} className="border rounded px-2 py-1 w-20" />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">P (mg/kg)</label>
                  <input type="number" value={manualP} onChange={e => setManualP(e.target.value)} className="border rounded px-2 py-1 w-20" />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">K (mg/kg)</label>
                  <input type="number" value={manualK} onChange={e => setManualK(e.target.value)} className="border rounded px-2 py-1 w-20" />
                </div>
              </div>
            )}
          </div>
        )}
        {!predictionResult ? (
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-gray-700 mb-4">
              {selectedRegion.length > 0
                ? `Region selected. Click "Analyze" to predict crop yield.`
                : "Draw a polygon on the map to select a region for analysis."}
            </p>
            <div className="flex space-x-2">
              <button
                onClick={handleSubmitInternal}
                disabled={selectedRegion.length === 0}
                className={`px-4 py-2 rounded-md ${
                  selectedRegion.length > 0
                    ? "bg-green-600 text-white hover:bg-green-700"
                    : "bg-gray-300 text-gray-500 cursor-not-allowed"
                }`}
              >
                Analyze
              </button>

              <button
                onClick={onReset}
                disabled={selectedRegion.length === 0}
                className={`px-4 py-2 rounded-md ${
                  selectedRegion.length > 0
                    ? "bg-gray-200 text-gray-700 hover:bg-gray-300"
                    : "bg-gray-100 text-gray-400 cursor-not-allowed"
                }`}
              >
                Reset
              </button>

              <button
                onClick={() => setShowHistory(!showHistory)}
                className="px-4 py-2 rounded-md bg-blue-50 text-blue-600 hover:bg-blue-100 flex items-center"
              >
                <Clock className="w-4 h-4 mr-1" />
                History
              </button>
            </div>

            {error && <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">{error}</div>}

            {showHistory && (
              <div className="mt-4 border rounded-md p-3">
                <h3 className="font-medium mb-2">Saved Regions</h3>
                {savedRegions.length === 0 ? (
                  <p className="text-gray-500 text-sm">No saved regions found</p>
                ) : (
                  <ul className="space-y-2">
                    {savedRegions.map((region) => (
                      <li key={region.id} className="flex justify-between items-center">
                        <span className="text-sm">Region {region.id.split("-")[1]}</span>
                        <button
                          onClick={() => onLoadRegion(region.coordinates)}
                          className="text-blue-600 text-sm hover:underline"
                        >
                          Load
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col h-full">
            <div className="flex border-b">
              <button
                className={`px-4 py-2 ${activeTab === "imagery" ? "tab-active text-green-600 border-b-2 border-green-600" : "text-gray-700"}`}
                onClick={() => setActiveTab("imagery")}
              >
                Imagery
              </button>
              <button
                className={`px-4 py-2 ${activeTab === "crops" ? "tab-active text-green-600 border-b-2 border-green-600" : "text-gray-700"}`}
                onClick={() => setActiveTab("crops")}
              >
                Crop Predictions
              </button>
              <button
                className={`px-4 py-2 ${activeTab === "data" ? "tab-active text-green-600 border-b-2 border-green-600" : "text-gray-700"}`}
                onClick={() => setActiveTab("data")}
              >
                Data
              </button>
              <button
                className={`px-4 py-2 ${activeTab === "interpretation" ? "tab-active text-green-600 border-b-2 border-green-600" : "text-gray-700"}`}
                onClick={() => setActiveTab("interpretation")}
              >
                AI Analysis
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-2">
              {activeTab === "imagery" && (
                <div>
                  <h3 className="font-medium mb-2">
                    {predictionResult.image ? "Satellite Imagery & Yield Prediction" : "No imagery available"}
                  </h3>
                  {predictionResult.image ? (
                    <img
                      src={`data:image/png;base64,${predictionResult.image}`}
                      alt="Satellite imagery and yield prediction"
                      className="w-full rounded-lg"
                    />
                  ) : (
                    <p className="text-gray-500">No imagery available</p>
                  )}
                </div>
              )}

              {activeTab === "crops" && (
                <div>
                  <h3 className="font-medium mb-2">Predicted Suitable Crops</h3>
                  <div className="space-y-4">
                    {predictionResult.prediction.predictions.map((pred, index) => (
                      <div
                        key={index}
                        className={`p-3 rounded-lg ${
                          pred.probability > 0.7
                            ? "bg-green-100 border border-green-300"
                            : pred.probability > 0.4
                              ? "bg-yellow-100 text-yellow-800"
                              : "bg-red-100 text-red-800"
                        }`}
                      >
                        <div className="flex justify-between items-center">
                          <h4 className="font-medium">
                            {index + 1}. {pred.class}
                          </h4>
                          <span
                            className={`text-sm px-2 py-1 rounded-full ${
                              pred.probability > 0.7
                                ? "bg-green-100 text-green-800"
                                : pred.probability > 0.4
                                  ? "bg-yellow-100 text-yellow-800"
                                  : "bg-red-100 text-red-800"
                            }`}
                          >
                            {(pred.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Feature importance visualization */}
                  <h3 className="font-medium mt-6 mb-3 flex items-center">
                    <span>Feature Importance</span>
                    <InfoTooltip content="The most important factors influencing crop predictions according to the ML model" />
                  </h3>

                  {predictionResult.prediction.feature_importances && predictionResult.prediction.feature_importances.length > 0 ? (
                    <div className="space-y-4">
                      {predictionResult.prediction.feature_importances.map((crop, idx) => (
                        <div key={idx} className="bg-gray-50 p-3 rounded-lg">
                          <h4 className="font-medium text-sm mb-2">{crop.crop}</h4>
                          {crop.features.map((feature, fidx) => (
                            <div key={fidx} className="mb-2">
                              <div className="flex justify-between text-xs mb-1">
                                <span>{feature.feature}</span>
                                <span>{feature.importance.toFixed(4)}</span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${feature.importance > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                                  style={{
                                    width: `${Math.min(Math.abs(feature.importance * 100), 100)}%`,
                                    marginLeft: feature.importance < 0 ? 'auto' : 0
                                  }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="bg-gray-50 p-3 rounded-lg flex items-center">
                      {backgroundTaskStatus === 'processing' ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                          <p>Analyzing feature importance, please wait...</p>
                        </>
                      ) : backgroundTaskStatus === 'error' ? (
                        <p className="text-red-600">Error generating feature importance</p>
                      ) : (
                        <p className="text-gray-500">No feature importance data available</p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {activeTab === "data" && (
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center mb-1">
                      <h3 className="font-medium">NDVI Data</h3>
                      <InfoTooltip content="Normalized Difference Vegetation Index - A measure of plant health and density" />
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <p>
                        Mean NDVI: <span className="font-medium">{predictionResult.ndvi_mean.toFixed(4)}</span>
                      </p>
w                      {typeof predictionResult.mean_evi === 'number' && (
                        <p>
                          Mean EVI: <span className="font-medium">{predictionResult.mean_evi.toFixed(4)}</span>
                        </p>
                      )}
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium mb-1">Environmental Data</h3>
                    <div className="bg-gray-50 p-3 rounded-lg grid grid-cols-2 gap-2">
                      <div>
                        <p className="text-sm text-gray-500">Temperature</p>
                        <p>{predictionResult.temperature_C || "N/A"} °C</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Rainfall</p>
                        <p>{predictionResult.rainfall_mm || "N/A"} mm</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Soil Moisture</p>
                        <p>{predictionResult.soil_moisture || "N/A"} %</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Predicted Yield</p>
                        <p>
                          {predictionResult.predicted_yield
                            ? (predictionResult.predicted_yield * 100).toFixed(1) + "%"
                            : "N/A"}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === "interpretation" && (
                <div>
                  <h3 className="font-medium mb-2">AI Interpretation</h3>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    {predictionResult.prediction.ai_interpretation === "AI interpretation loading..." ? (
                      <div className="flex items-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                        <p>Generating AI interpretation, please wait...</p>
                      </div>
                    ) : (
                      <p className="whitespace-pre-line">{predictionResult.prediction.ai_interpretation}</p>
                    )}
                  </div>

                  <h3 className="font-medium mt-6 mb-2">Model Explanations</h3>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    {predictionResult.prediction.explanations === "Detailed explanations will be available shortly..." ? (
                      <div className="flex items-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                        <p>Generating detailed explanations, please wait...</p>
                      </div>
                    ) : (
                      <pre className="whitespace-pre-line text-sm">{predictionResult.prediction.explanations}</pre>
                    )}
                  </div>
                </div>
              )}

              {/* Show task progress indicator if any background task is running */}
              {backgroundTaskStatus && backgroundTaskStatus !== 'complete' && (
                <div className="mt-4 bg-blue-50 p-3 rounded-lg">
                  <div className="flex items-center mb-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                    <h3 className="font-medium text-blue-800">Processing in background</h3>
                  </div>
                  <p className="text-sm text-blue-600">
                    {backgroundTaskStatus === 'processing' ? 
                      'Generating advanced analysis and interpretations...' : 
                      backgroundTaskStatus === 'lime_complete' ? 
                      'Feature importance analysis complete. Generating AI interpretation...' :
                      'Almost done...'}
                  </p>
                  <div className="w-full bg-blue-200 rounded-full h-2 mt-2">
                    <div 
                      className="h-2 rounded-full bg-blue-600" 
                      style={{ 
                        width: backgroundTaskStatus === 'processing' ? '33%' : 
                               backgroundTaskStatus === 'lime_complete' ? '66%' : '90%'
                      }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-between mt-4 pt-2 border-t">
              <button
                onClick={onReset}
                className="px-4 py-2 rounded-md bg-gray-200 text-gray-700 hover:bg-gray-300 flex items-center"
              >
                <RefreshCw className="w-4 h-4 mr-1" />
                New Analysis
              </button>

              <button
                onClick={handleExportPDF}
                className="px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 flex items-center"
              >
                <Download className="w-4 h-4 mr-1" />
                Export PDF
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
