@echo off
setlocal

:: This is a multi-purpose script. If it's called with the argument ":waitForAndOpen",
:: it will jump to that section. Otherwise, it runs the main logic.
if "%1"==":waitForAndOpen" goto :waitForAndOpen

:: ============================================================================
:: MAIN SCRIPT LOGIC
:: ============================================================================

:: --- Configuration ---
SET IMAGE_NAME=f5tts:latest
SET TAR_FILE="f5tts-app-preloaded_2025-10-21.tar"
SET CONTAINER_NAME=f5tts-gradio-instance
SET APP_PORT=7860
SET APP_URL=http://127.0.0.1:%APP_PORT%
:: ---------------------

:: This magic line forces the script to run from its own directory
cd /d "%~dp0"

echo Checking for local Docker image: %IMAGE_NAME%
docker image inspect %IMAGE_NAME% >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
  echo Image not found locally. Loading from %TAR_FILE%... This may take a while.
  docker load -i %TAR_FILE%
) ELSE (
  echo Image already exists locally. Skipping load.
)

echo.
echo --- Preparing to Launch ---
echo Cleaning up any previous container instance...
:: Suppress errors in case the container doesn't exist
docker stop %CONTAINER_NAME% >nul 2>nul
docker rm %CONTAINER_NAME% >nul 2>nul

echo.
echo Starting the application. A browser window will open automatically when it's ready.
echo To stop the application, simply CLOSE THIS WINDOW.
echo.

:: Launch the browser-checker task in the background IN THE SAME WINDOW.
:: It calls this very script file with a special argument to run the subroutine.
start "" /B cmd /c call "%~f0" :waitForAndOpen %CONTAINER_NAME% %APP_URL%

:: Run the main container command in the FOREGROUND. This is a blocking call.
:: --rm ensures the container is removed when this command is terminated (by closing the window).
docker run --gpus all -it --rm --name %CONTAINER_NAME% -p %APP_PORT%:%APP_PORT% -v "%CD%:/app/data" %IMAGE_NAME%

echo.
echo Docker container has been stopped.
pause
goto :eof

:: ============================================================================
:: BACKGROUND SUBROUTINE (Do not call directly)
:: ============================================================================
:waitForAndOpen
    SET "TARGET_CONTAINER=%~2"
    SET "TARGET_URL=%~3"

    echo [Background Task] Waiting for application to initialize...

    :log_check_loop
    :: Wait for 2 seconds before checking.
    timeout /t 2 /nobreak >nul

    :: Check if the container is still running before checking its logs.
    docker ps -f "name=%TARGET_CONTAINER%" | findstr "%TARGET_CONTAINER%" >nul
    IF %ERRORLEVEL% NEQ 0 (
        echo [Background Task] Container stopped unexpectedly. Aborting.
        exit /B
    )

    :: Check the container's logs for the "ready" message.
    docker logs %TARGET_CONTAINER% 2>&1 | findstr /C:"* Running on local URL:" >nul
    IF %ERRORLEVEL% NEQ 0 (
        goto log_check_loop
    )

    echo [Background Task] Application is ready! Opening URL: %TARGET_URL%
    start "" "%TARGET_URL%"
    exit /B