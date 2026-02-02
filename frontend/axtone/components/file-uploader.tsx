'use client'

import React from "react"

import { useState, useRef } from 'react'
import { Upload, Music } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface FileUploaderProps {
  onFileSelect: (file: File) => void
}

export default function FileUploader({ onFileSelect }: FileUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('audio/')) {
        onFileSelect(file)
      }
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      onFileSelect(files[0])
    }
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 p-12 text-center cursor-pointer
        ${isDragOver 
          ? 'border-cyan-400 bg-cyan-400/5' 
          : 'border-slate-700 bg-slate-900/50 hover:border-slate-600'
        }`}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/mp3,audio/wav,audio/mpeg,.mp3,.wav"
        onChange={handleFileInput}
        className="hidden"
      />

      <div className="space-y-4">
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-gradient-to-br from-cyan-400/20 to-blue-600/20 rounded-full flex items-center justify-center border border-cyan-400/30">
            <Upload className="w-8 h-8 text-cyan-400" />
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-white">
            Drop your audio file here
          </h3>
          <p className="text-slate-400">
            or click to select a file
          </p>
          <p className="text-sm text-slate-500">
            Supports MP3 and WAV formats (Max 50MB)
          </p>
        </div>

        <Button 
          className="mt-6 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0"
          size="lg"
        >
          <Music className="w-4 h-4 mr-2" />
          Choose File
        </Button>
      </div>
    </div>
  )
}
