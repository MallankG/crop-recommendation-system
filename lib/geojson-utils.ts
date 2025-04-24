export interface GeoJSONFeature {
  type: string
  properties: Record<string, any>
  geometry: {
    type: string
    coordinates: any[]
  }
}

export interface GeoJSONData {
  type: string
  features: GeoJSONFeature[]
}

export function parseGeoJSON(fileContent: string): {
  regions: { name: string; coordinates: number[][] }[]
  error?: string
} {
  try {
    const data: GeoJSONData = JSON.parse(fileContent)

    if (data.type !== "FeatureCollection") {
      return {
        regions: [],
        error: "Invalid GeoJSON: Must be a FeatureCollection",
      }
    }

    const regions: { name: string; coordinates: number[][] }[] = []

    data.features.forEach((feature, index) => {
      if (feature.geometry.type === "Polygon") {
        // Get the outer ring of the polygon
        const coordinates = feature.geometry.coordinates[0]
        const name = feature.properties.name || `Region ${index + 1}`

        regions.push({
          name,
          coordinates,
        })
      }
    })

    if (regions.length === 0) {
      return {
        regions: [],
        error: "No valid polygon features found in the GeoJSON file",
      }
    }

    return { regions }
  } catch (error) {
    return {
      regions: [],
      error: `Failed to parse GeoJSON: ${error instanceof Error ? error.message : "Unknown error"}`,
    }
  }
}
