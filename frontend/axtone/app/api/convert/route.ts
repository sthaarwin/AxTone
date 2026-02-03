import { NextRequest, NextResponse } from 'next/server';

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    // Forward the request to the Python FastAPI backend
    const response = await fetch(`${PYTHON_API_URL}/api/convert`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || 'Failed to process audio file' },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { 
        error: 'Failed to connect to processing server. Make sure the Python API is running on port 8000.',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  // Health check endpoint
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/health`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { status: 'error', message: 'Python API not reachable' },
      { status: 503 }
    );
  }
}
