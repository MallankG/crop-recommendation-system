"use client"

import { useEffect, useRef, useState } from "react"
import L from "leaflet"
import "leaflet/dist/leaflet.css"
import "leaflet-draw/dist/leaflet.draw.css"
import { MapContainer, TileLayer, FeatureGroup } from "react-leaflet"
import { EditControl } from "react-leaflet-draw"

interface MapComponentProps {
  onRegionSelect: (coordinates: number[][]) => void
  selectedRegion: number[][]
}

export default function MapComponent({ onRegionSelect, selectedRegion }: MapComponentProps) {
  const [map, setMap] = useState<L.Map | null>(null)
  const featureGroupRef = useRef<L.FeatureGroup | null>(null)
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  // Fix Leaflet icon issues
  useEffect(() => {
    if (isClient) {
      delete (L.Icon.Default.prototype as any)._getIconUrl
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
        iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
        shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
      })
    }
  }, [isClient])

  // Clear existing layers and add the selected region if it exists
  useEffect(() => {
    if (map && featureGroupRef.current) {
      featureGroupRef.current.clearLayers()

      if (selectedRegion.length > 0) {
        const polygon = L.polygon(selectedRegion.map((coord) => [coord[1], coord[0]]))
        featureGroupRef.current.addLayer(polygon)

        // Fit the map to the polygon bounds
        map.fitBounds(polygon.getBounds())
      }
    }
  }, [map, selectedRegion])

  const handleCreated = (e: any) => {
    const layer = e.layer

    if (layer instanceof L.Polygon) {
      const latLngs = layer.getLatLngs()[0]
      const coordinates = (latLngs as L.LatLng[]).map((latLng) => [latLng.lng, latLng.lat])
      onRegionSelect(coordinates)
    }
  }

  const handleDeleted = () => {
    onRegionSelect([])
  }

  const handleEdited = (e: any) => {
    const layers = e.layers
    layers.eachLayer((layer: L.Polygon) => {
      const latLngs = layer.getLatLngs()[0]
      const coordinates = (latLngs as L.LatLng[]).map((latLng) => [latLng.lng, latLng.lat])
      onRegionSelect(coordinates)
    })
  }

  return (
    <MapContainer center={[20, 0]} zoom={3} style={{ height: "100%", width: "100%" }} whenCreated={setMap}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FeatureGroup ref={featureGroupRef}>
        <EditControl
          position="topright"
          onCreated={handleCreated}
          onDeleted={handleDeleted}
          onEdited={handleEdited}
          draw={{
            rectangle: false,
            circle: false,
            circlemarker: false,
            marker: false,
            polyline: false,
            polygon: {
              allowIntersection: false,
              drawError: {
                color: "#e1e100",
                message: "<strong>Error:</strong> Polygon edges cannot cross!",
              },
              shapeOptions: {
                color: "#4CAF50",
                fillOpacity: 0.2,
              },
            },
          }}
        />
      </FeatureGroup>
    </MapContainer>
  )
}
