@echo off
echo Running Docker container 'f5tts'...
docker run -it -p 7860:7860 --gpus all f5tts
if %ERRORLEVEL% NEQ 0 (
    echo Docker run failed. Please check for errors.
    pause
    exit /b %ERRORLEVEL%
)
echo Docker container 'f5tts' stopped or exited successfully.
pause
