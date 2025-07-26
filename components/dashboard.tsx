"use client"

import { useState, useEffect } from "react"
import dynamic from "next/dynamic"
import AnalysisPanel from "./analysis-panel"
import Header from "./header"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { PredictionResult } from "@/lib/types"
import LoadingOverlay from "./loading-overlay"
import { parseGeoJSON } from "@/lib/geojson-utils"
import { v4 as uuidv4 } from "uuid"

// Dynamically import the MapComponent to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import("./map-component"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-gray-100">
      <div className="loading-spinner"></div>
    </div>
  ),
})

export default function Dashboard() {
  const [selectedRegion, setSelectedRegion] = useState<number[][]>([])
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [savedRegions, setSavedRegions] = useState<{ id: string; coordinates: number[][] }[]>([])
  const [activeTab, setActiveTab] = useState("single")
  const [backgroundTaskId, setBackgroundTaskId] = useState<string | null>(null)
  const [backgroundTaskInterval, setBackgroundTaskInterval] = useState<NodeJS.Timeout | null>(null)
  const [backgroundTaskStatus, setBackgroundTaskStatus] = useState<'processing' | 'lime_complete' | 'complete' | 'error' | null>(null)

  // Clean up the polling interval when component unmounts
  useEffect(() => {
    return () => {
      if (backgroundTaskInterval) {
        clearInterval(backgroundTaskInterval);
      }
    };
  }, [backgroundTaskInterval]);

  // Load saved regions from local storage on component mount
  useEffect(() => {
    const savedRegionsData = localStorage.getItem("savedRegions")
    if (savedRegionsData) {
      try {
        setSavedRegions(JSON.parse(savedRegionsData))
      } catch (err) {
        console.error("Error parsing saved regions:", err)
      }
    }
  }, [])

  const handleRegionSelect = (coordinates: number[][]) => {
    setSelectedRegion(coordinates)
  }

  const handleSubmit = async () => {
    if (selectedRegion.length === 0) {
      setError("Please select a region on the map first")
      return
    }
    setIsLoading(true)
    setError(null)
    setPredictionResult(null)
    setBackgroundTaskStatus(null)
    setBackgroundTaskId(null)
    if (backgroundTaskInterval) {
      clearInterval(backgroundTaskInterval)
      setBackgroundTaskInterval(null)
    }
    try {
      setError("Processing request. This may take some time for large regions...")
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          roi: selectedRegion
        })
      });
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`)
      }
      const data = await response.json()
      // Check if we got a valid response
      if (!data || (data.error && !data.ndvi_mean)) {
        throw new Error(data.error || "Invalid response from server");
      }

      // Make sure prediction has required fields
      const prediction = data.prediction || {
        predictions: [],
        explanations: "No detailed explanations available.",
        ai_interpretation: "Interpretation not available.",
        feature_importances: []
      };

      // If we have at least the NDVI value, show what we have
      setPredictionResult({
        ...data,
        prediction: prediction
      })

      // Save the region to local storage
      const regionId = `region-${Date.now()}`
      const newSavedRegions = [...savedRegions, { id: regionId, coordinates: selectedRegion }]
      setSavedRegions(newSavedRegions)
      localStorage.setItem("savedRegions", JSON.stringify(newSavedRegions))

      // Clear any error messages if we got results
      setError(null)

      // Start polling for background task updates
      if (data.task_id) {
        startPollingBackgroundTask(data.task_id);
      }
    } catch (err: any) {
      console.error("Error submitting prediction:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const startPollingBackgroundTask = (taskId: string) => {
    // Clear any existing interval
    if (backgroundTaskInterval) {
      clearInterval(backgroundTaskInterval);
    }

    setBackgroundTaskId(taskId);
    setBackgroundTaskStatus('processing');

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/task-status/${taskId}`);
        if (!response.ok) {
          if (response.status === 404) {
            console.error('Background task not found');
            clearInterval(interval);
            setBackgroundTaskInterval(null);
            return;
          }
          throw new Error(`Failed to fetch task status: ${response.status}`);
        }

        const data = await response.json();
        setBackgroundTaskStatus(data.status);

        // Update prediction result with LIME and Groq data when available
        if (data.status === 'lime_complete' || data.status === 'complete') {
          if (predictionResult) {
            setPredictionResult(prev => {
              if (!prev) return prev;
              return {
                ...prev,
                prediction: {
                  ...prev.prediction,
                  explanations: data.result.explanations || prev.prediction.explanations,
                  ai_interpretation: data.result.ai_interpretation || prev.prediction.ai_interpretation,
                  feature_importances: data.result.feature_importances || prev.prediction.feature_importances || []
                }
              };
            });
          }
        }

        // Stop polling when complete or error
        if (data.status === 'complete' || data.status === 'error') {
          clearInterval(interval);
          setBackgroundTaskInterval(null);
        }
      } catch (error) {
        console.error('Error polling background task:', error);
      }
    }, 2000); // Poll every 2 seconds

    setBackgroundTaskInterval(interval);
  };

  const handleReset = () => {
    setSelectedRegion([])
    setPredictionResult(null)
    setError(null)
  }

  const handleLoadRegion = (coordinates: number[][]) => {
    setSelectedRegion(coordinates)
    setPredictionResult(null)
  }

  return (
    <div className="flex flex-col h-screen">
      <Header />

      <div className="flex flex-col md:flex-row flex-1 overflow-hidden">
        <div className="w-full md:w-2/3 h-[50vh] md:h-full relative">
          <MapComponent onRegionSelect={handleRegionSelect} selectedRegion={selectedRegion} />
          {isLoading && <LoadingOverlay />}
        </div>

        <div className="w-full md:w-1/3 h-[50vh] md:h-full overflow-y-auto bg-white shadow-md p-4">
          <AnalysisPanel
            selectedRegion={selectedRegion}
            predictionResult={predictionResult}
            onSubmit={handleSubmit}
            onReset={handleReset}
            error={error}
            savedRegions={savedRegions}
            onLoadRegion={handleLoadRegion}
            backgroundTaskStatus={backgroundTaskStatus}
          />
        </div>
      </div>
    </div>
  )
}
