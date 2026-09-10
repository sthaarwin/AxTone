'use client'

import React from "react"

import { useState, useRef } from 'react'
import { Upload, Music, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface FileUploaderProps {
  onFileSelect: (file: File) => void
}

const ACCEPTED_EXTENSIONS = '.mp3,.wav,.flac,.ogg,.m4a,.aac'
const ACCEPTED_MIME = 'audio/mp3,audio/mpeg,audio/wav,audio/flac,audio/ogg,audio/mp4,audio/x-m4a,audio/aac,audio/*'
const MAX_FILE_SIZE_MB = 50
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

export default function FileUploader({ onFileSelect }: FileUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [sizeError, setSizeError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const validateAndSelect = (file: File) => {
    setSizeError('')

    // Client-side file size guard (mirrors the 50 MB server limit)
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setSizeError(
        `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). ` +
        `Maximum allowed size is ${MAX_FILE_SIZE_MB} MB.`
      )
      return
    }

    onFileSelect(file)
  }

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
    setSizeError('')

    const files = e.dataTransfer.files
    if (files.length > 0) {
      validateAndSelect(files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSizeError('')
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      validateAndSelect(files[0])
    }
    // Reset input so the same file can be re-selected after an error
    e.currentTarget.value = ''
  }

  return (
    <div className="space-y-3">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 p-12 text-center cursor-pointer
          ${isDragOver
            ? 'border-cyan-400 bg-cyan-400/5'
            : sizeError
              ? 'border-red-500 bg-red-500/5'
              : 'border-slate-700 bg-slate-900/50 hover:border-slate-600'
          }`}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={`${ACCEPTED_MIME},${ACCEPTED_EXTENSIONS}`}
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
              Supports MP3, WAV, FLAC, OGG, M4A, AAC &nbsp;·&nbsp; Max {MAX_FILE_SIZE_MB} MB
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

      {/* Client-side size error */}
      {sizeError && (
        <div className="flex items-start gap-2 rounded-lg border border-red-800 bg-red-900/20 px-4 py-3">
          <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
          <p className="text-sm text-red-300">{sizeError}</p>
        </div>
      )}
    </div>
  )
}
