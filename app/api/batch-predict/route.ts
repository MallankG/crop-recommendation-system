import { type NextRequest, NextResponse } from "next/server"
import { v4 as uuidv4 } from "uuid"

// This would be a database in a real application
const batchJobs: Record<string, any> = {}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()
    const { regions, selectedCrop } = data

    if (!regions || !Array.isArray(regions) || regions.length === 0) {
      return NextResponse.json({ error: "No valid regions provided" }, { status: 400 })
    }

    // Create a batch job ID
    const batchId = uuidv4()

    // Initialize batch job status
    batchJobs[batchId] = {
      id: batchId,
      regions: regions.map((region, index) => ({
        id: uuidv4(),
        name: region.name || `Region ${index + 1}`,
        coordinates: region.coordinates,
        status: "pending",
        createdAt: new Date().toISOString(),
      })),
      selectedCrop,
      createdAt: new Date().toISOString(),
    }

    // Start processing in the background
    processBatchJob(batchId, selectedCrop)

    return NextResponse.json({
      batchId,
      message: "Batch processing started",
      jobCount: regions.length,
    })
  } catch (error) {
    console.error("Error in batch-predict:", error)
    return NextResponse.json({ error: "Failed to process batch prediction request" }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const batchId = searchParams.get("batchId")

  if (!batchId) {
    return NextResponse.json({ error: "Batch ID is required" }, { status: 400 })
  }

  const batchJob = batchJobs[batchId]

  if (!batchJob) {
    return NextResponse.json({ error: "Batch job not found" }, { status: 404 })
  }

  return NextResponse.json(batchJob)
}

// This would be a background job in a real application
async function processBatchJob(batchId: string, selectedCrop: string | null) {
  const batchJob = batchJobs[batchId]

  if (!batchJob) return

  // Process each region sequentially
  for (const region of batchJob.regions) {
    try {
      // Update status to processing
      region.status = "processing"

      // Call the Flask backend
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          roi: region.coordinates,
          crop: selectedCrop || undefined,
        }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`)
      }

      const result = await response.json()

      // Update region with result
      region.status = "completed"
      region.result = result
      region.completedAt = new Date().toISOString()
    } catch (error) {
      console.error(`Error processing region ${region.id}:`, error)
      region.status = "failed"
      region.error = error instanceof Error ? error.message : "Unknown error"
    }
  }
}
