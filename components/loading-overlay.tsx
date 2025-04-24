export default function LoadingOverlay() {
  return (
    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
      <div className="bg-white p-6 rounded-lg shadow-lg flex flex-col items-center">
        <div className="loading-spinner mb-4"></div>
        <p className="text-gray-700">Analyzing region...</p>
        <p className="text-gray-500 text-sm mt-2">This may take a few moments</p>
      </div>
    </div>
  )
}
