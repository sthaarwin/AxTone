import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    // Get API URL from environment variable
    const PYTHON_API_URL = process.env.PYTHON_API_URL || process.env.NEXT_PUBLIC_PYTHON_API_URL;
    
    if (!PYTHON_API_URL) {
      throw new Error('API URL not configured');
    }

    console.log('Connecting to:', PYTHON_API_URL);
    
    // Forward the request to the Python FastAPI backend with extended timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minute timeout for server wake-up
    
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/convert`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      console.log('✅ Response received:', response.status);

      if (!response.ok) {
        const error = await response.json();
        console.error('Backend error:', error);
        return NextResponse.json(
          { error: error.detail || 'Conversion failed' },
          { status: response.status }
        );
      }

      const data = await response.json();
      console.log('✅ Conversion successful');
      return NextResponse.json(data);
      
    } catch (fetchError) {
      clearTimeout(timeoutId);
      
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        throw new Error('Request timed out. Please try again.');
      }
      
      throw new Error(`Cannot connect to server: ${fetchError instanceof Error ? fetchError.message : 'Network error'}`);
    }
    
  } catch (error) {
    console.error('API route error:', error);
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to connect to processing server' 
      },
      { status: 500 }
    );
  }
}