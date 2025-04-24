export default function Header() {
  return (
    <header className="bg-green-600 text-white p-4 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Crop Yield Prediction System</h1>
          <p className="text-green-100">Analyze regions for optimal crop selection and yield estimation</p>
        </div>
        <div className="hidden md:block">
          <div className="flex items-center space-x-2">
            <span className="bg-green-700 px-3 py-1 rounded-full text-sm">Powered by Earth Engine & ML</span>
          </div>
        </div>
      </div>
    </header>
  )
}
