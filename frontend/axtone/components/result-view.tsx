'use client'

import { useState } from 'react'
import { Play, Pause, Download, FileDown, Music as MusicIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import WaveformVisualizer from './waveform-visualizer'

interface ResultViewProps {
  fileName: string
  isPlaying: boolean
  onPlayingChange: (playing: boolean) => void
}

const SAMPLE_TAB = `e|-------|-------|-5-7-5-|-7-9-7-|
B|---1---|-3-1---|-5-7-5-|-5-7-5-|
G|-2---2-|-----2-|-6-8-6-|-6-8-6-|
D|-------|-------|-7-9-7-|-7-9-7-|
A|-------|-------|-7-9-7-|-7-9-7-|
E|-------|-------|-------|-------|`

export default function ResultView({ 
  fileName, 
  isPlaying, 
  onPlayingChange 
}: ResultViewProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(SAMPLE_TAB)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownloadTab = () => {
    const element = document.createElement('a')
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(SAMPLE_TAB))
    element.setAttribute('download', fileName.replace(/\.[^/.]+$/, '') + '.txt')
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Your Tab</h2>
          <p className="text-slate-400 text-sm mt-1">{fileName}</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => onPlayingChange(!isPlaying)}
            className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0"
            size="lg"
          >
            {isPlaying ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Play
              </>
            )}
          </Button>
          <Button
            onClick={handleDownloadTab}
            variant="outline"
            size="lg"
            className="border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent"
          >
            <FileDown className="w-4 h-4 mr-2" />
            Download Tab
          </Button>
        </div>
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Waveform Visualizer */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
          <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <MusicIcon className="w-4 h-4 text-cyan-400" />
            Waveform
          </h3>
          <WaveformVisualizer isPlaying={isPlaying} />
        </div>

        {/* Guitar Tab */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm overflow-x-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <MusicIcon className="w-4 h-4 text-cyan-400" />
              Tablature
            </h3>
            <Button
              onClick={handleCopy}
              variant="ghost"
              size="sm"
              className="text-slate-400 hover:text-slate-200 hover:bg-slate-800 text-xs"
            >
              {copied ? 'Copied!' : 'Copy'}
            </Button>
          </div>
          <div className="font-mono text-sm text-slate-300 leading-relaxed whitespace-pre-wrap break-words max-w-full">
            {SAMPLE_TAB}
          </div>
        </div>
      </div>

      {/* Download Options */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
        <h3 className="text-sm font-semibold text-slate-200 mb-4">Export Options</h3>
        <div className="flex flex-col sm:flex-row gap-3">
          <Button
            onClick={handleDownloadTab}
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-100"
          >
            <FileDown className="w-4 h-4 mr-2" />
            Download as .txt
          </Button>
          <Button
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-100"
          >
            <Download className="w-4 h-4 mr-2" />
            Download as MIDI
          </Button>
          <Button
            variant="outline"
            className="flex-1 border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent"
          >
            Share Link
          </Button>
        </div>
      </div>
    </div>
  )
}
