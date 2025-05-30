@echo off
echo Building Docker image 'f5tts'...
docker build -t f5tts .
if %ERRORLEVEL% NEQ 0 (
    echo Docker build failed. Please check for errors.
    pause
    exit /b %ERRORLEVEL%
)
echo Docker image 'f5tts' built successfully.
pause
