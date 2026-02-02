'use client'

import { useEffect, useState } from 'react'
import { Loader } from 'lucide-react'

interface ProcessingStateProps {
  fileName: string
}

interface ProcessingStep {
  id: string
  label: string
  completed: boolean
}

export default function ProcessingState({ fileName }: ProcessingStateProps) {
  const [steps, setSteps] = useState<ProcessingStep[]>([
    { id: 'extract', label: 'Extracting Frequencies', completed: false },
    { id: 'graph', label: 'Building Fretboard Graph', completed: false },
    { id: 'dijkstra', label: 'Running Dijkstra Optimization', completed: false },
  ])

  useEffect(() => {
    // Animate steps completion
    const timings = [800, 2000, 3200]
    
    timings.forEach((timing, index) => {
      setTimeout(() => {
        setSteps(prev => 
          prev.map((step, i) => 
            i === index ? { ...step, completed: true } : step
          )
        )
      }, timing)
    })
  }, [])

  return (
    <div className="flex flex-col items-center justify-center min-h-[600px] space-y-12">
      {/* Animated Logo */}
      <div className="relative w-24 h-24">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg blur-xl opacity-30 animate-pulse" />
        <div className="relative w-full h-full bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center border border-cyan-400/50">
          <div className="text-4xl font-bold text-white">♪</div>
        </div>
      </div>

      {/* File Info */}
      <div className="text-center space-y-2">
        <p className="text-slate-400 text-sm">Processing</p>
        <p className="text-white font-semibold truncate max-w-xs">{fileName}</p>
      </div>

      {/* Processing Steps */}
      <div className="space-y-4 w-full max-w-md">
        {steps.map((step) => (
          <div key={step.id} className="flex items-center gap-4">
            <div className="relative w-8 h-8 flex-shrink-0">
              {step.completed ? (
                <div className="w-full h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              ) : (
                <div className="w-full h-full rounded-full border-2 border-slate-700 flex items-center justify-center">
                  <Loader className="w-4 h-4 text-cyan-400 animate-spin" />
                </div>
              )}
            </div>
            <span className={`text-sm font-medium ${step.completed ? 'text-slate-300' : 'text-slate-400'}`}>
              {step.label}
            </span>
          </div>
        ))}
      </div>

      {/* Progress Info */}
      <div className="text-center">
        <p className="text-slate-500 text-sm">
          Analyzing audio and building optimal fretboard path...
        </p>
      </div>
    </div>
  )
}
