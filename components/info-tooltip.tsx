import { HelpCircle } from "lucide-react"

interface InfoTooltipProps {
  content: string
}

export default function InfoTooltip({ content }: InfoTooltipProps) {
  return (
    <div className="tooltip ml-1">
      <HelpCircle className="w-4 h-4 text-gray-400" />
      <span className="tooltip-text">{content}</span>
    </div>
  )
}
