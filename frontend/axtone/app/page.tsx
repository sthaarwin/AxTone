'use client'

import { useState, useRef } from 'react'
import FileUploader from '@/components/file-uploader'
import ProcessingState from '@/components/processing-state'
import ResultView from '@/components/result-view'
import InteractiveFretboard from '@/components/interactive-fretboard'
import SettingsPanel from '@/components/settings-panel'
import { Button } from '@/components/ui/button'
import { Music } from 'lucide-react'

type AppState = 'idle' | 'processing' | 'result' | 'error'

interface ConversionResult {
  tablature: string
  midi_base64?: string
  notes?: Array<{
    midi: number
    onset: number
    offset: number
    string: number
    fret: number
  }>
  stats: {
    total_notes: number
    pitch_range: {
      min: string
      max: string
    }
    avg_fret_movement: number
    avg_string_jumps: number
    tuning_used: string
  }
}

export default function Home() {
  const [state, setState] = useState<AppState>('idle')
  const [fileName, setFileName] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)
  const [result, setResult] = useState<ConversionResult | null>(null)
  const [error, setError] = useState<string>('')
  const [tuning, setTuning] = useState('standard')
  const [method, setMethod] = useState('pyin')
  
  // Ref for scrolling to fretboard
  const fretboardRef = useRef<HTMLDivElement>(null)

  const handleFileUpload = async (file: File) => {
    setFileName(file.name)
    setState('processing')
    setError('')
    
    try {
      // Create FormData
      const formData = new FormData()
      formData.append('file', file)
      formData.append('method', method)
      formData.append('tuning', tuning)
      formData.append('min_duration', '0.1')
      formData.append('detailed', 'true')
      formData.append('preprocess', 'false')
      
      // Call the Next.js API route (which forwards to Python)
      const response = await fetch('/api/convert', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to convert audio')
      }
      
      const data = await response.json()
      setResult(data)
      setState('result')
      
    } catch (err) {
      console.error('Conversion error:', err)
      setError(err instanceof Error ? err.message : 'An error occurred during conversion')
      setState('error')
    }
  }

  const handlePlayToggle = (playing: boolean) => {
    setIsPlaying(playing)
    
    // When play is pressed, scroll to fretboard
    if (playing && fretboardRef.current) {
      fretboardRef.current.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
      })
    }
  }

  const handleReset = () => {
    setState('idle')
    setFileName('')
    setIsPlaying(false)
    setResult(null)
    setError('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center">
              <Music className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Axtone</h1>
              <p className="text-xs text-slate-400">AI Vocal to Guitar Tab</p>
            </div>
          </div>
          {state !== 'idle' && (
            <Button 
              onClick={handleReset}
              variant="outline"
              size="sm"
              className="border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent"
            >
              New Upload
            </Button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {state === 'idle' && (
          <div className="space-y-12">
            {/* Hero */}
            <div className="text-center space-y-4 py-8">
              <h2 className="text-4xl sm:text-5xl font-bold text-white text-balance">
                Convert Vocals to <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">Guitar Tabs</span>
              </h2>
              <p className="text-lg text-slate-400 max-w-2xl mx-auto">
                Upload a vocal recording and our AI will analyze the melody and generate accurate guitar tablature in seconds.
              </p>
            </div>

            {/* Settings and Upload */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <FileUploader onFileSelect={handleFileUpload} />
              </div>
              <div>
                <SettingsPanel 
                  tuning={tuning}
                  method={method}
                  onTuningChange={setTuning}
                  onMethodChange={setMethod}
                />
              </div>
            </div>
          </div>
        )}

        {state === 'processing' && (
          <ProcessingState fileName={fileName} />
        )}

        {state === 'error' && (
          <div className="max-w-2xl mx-auto">
            <div className="rounded-xl border border-red-800 bg-red-900/20 p-6 backdrop-blur-sm">
              <h3 className="text-xl font-bold text-red-400 mb-2">Conversion Failed</h3>
              <p className="text-red-300 mb-4">{error}</p>
              <Button 
                onClick={handleReset}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                Try Again
              </Button>
            </div>
          </div>
        )}

        {state === 'result' && result && (
          <>
            <ResultView 
              fileName={fileName}
              isPlaying={isPlaying}
              onPlayingChange={handlePlayToggle}
              tablature={result.tablature}
              stats={result.stats}
              midiBase64={result.midi_base64}
            />
            <div ref={fretboardRef} className="mt-12">
              <InteractiveFretboard 
                isPlaying={isPlaying} 
                onPlayingChange={handlePlayToggle}
                notes={result.notes || []}
              />
            </div>
          </>
        )}
      </main>
    </div>
  )
}
