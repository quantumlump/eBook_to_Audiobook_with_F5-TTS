@echo off
setlocal

:: This is a multi-purpose script. If it's called with the argument ":waitForAndOpen",
:: it will jump to that section. Otherwise, it runs the main logic.
if "%1"==":waitForAndOpen" goto :waitForAndOpen

:: ============================================================================
:: MAIN SCRIPT LOGIC
:: ============================================================================

:: --- Check for Docker and start if necessary ---
echo Checking if Docker Desktop is running...
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>NUL | find /I /N "Docker Desktop.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Docker Desktop is already running.
) else (
    echo Docker Desktop is not running. Starting it now...
    :: NOTE: The path below is the default installation path. If you installed
    :: Docker Desktop elsewhere, you will need to update this line.
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for the Docker engine to initialize... This may take a minute.
)

:waitForDockerLoop
:: Attempt a simple docker command until it succeeds.
docker info >nul 2>nul
if %errorlevel% neq 0 (
    echo Waiting for Docker...
    timeout /t 5 /nobreak >nul
    goto waitForDockerLoop
)
echo Docker is ready.
echo.
:: --- End Docker Check ---

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
  echo Image not found locally. Loading from %TAR_FILE%... This may take over 20 min.
  docker load -i %TAR_FILE%
) ELSE (
  echo Image already exists locally. Skipping load.
)

echo.
echo --- Preparing to Launch ---
echo Cleaning up any previous container instance...
:: Suppress errors in case the container doesn't exist. This is important for
:: cleaning up if the window was closed with the 'X' on a previous run.
docker stop %CONTAINER_NAME% >nul 2>nul
docker rm %CONTAINER_NAME% >nul 2>nul

echo.
echo Starting the application. A browser window will open automatically when it's ready.
echo.
echo ===================================================================
echo  IMPORTANT: To stop the application, press CTRL+C in this window.
echo             (Do not click the 'X' on the window title bar)
echo ===================================================================
echo.

:: Launch the browser-checker task in the background IN THE SAME WINDOW.
:: It calls this very script file with a special argument to run the subroutine.
start "" /B cmd /c call "%~f0" :waitForAndOpen %CONTAINER_NAME% %APP_URL%

:: Run the main container command in the FOREGROUND. This is a blocking call.
:: --rm ensures the container is removed when this command is terminated (by closing the window).
:: --stop-timeout 0 tells Docker to not wait for a graceful shutdown.
docker run --gpus all -it --rm --stop-timeout 0 --name %CONTAINER_NAME% -p %APP_PORT%:%APP_PORT% -v "%CD%:/app/data" %IMAGE_NAME%

echo.
echo ====================================================
echo      SHUTTING DOWN: Application has been stopped.
echo      Window will now close.
echo ====================================================

:: Add a very short, silent timeout to ensure the message is readable before closing.
timeout /t 2 /nobreak >nul

goto :eof

:: ============================================================================
:: BACKGROUND SUBROUTINE (Do not call directly)
:: ============================================================================
:waitForAndOpen
    SET "TARGET_CONTAINER=%~2"
    SET "TARGET_URL=%~3"
    SET "CHECK_INTERVAL=3"
    :: We need N+1 pings to get N seconds of wait time.
    SET /A PING_COUNT=%CHECK_INTERVAL% + 1

    echo.
    echo [Background Task] Monitoring container logs. Will open the URL when the app is ready.
    echo.

    :log_check_loop
    :: Wait for a few seconds before checking, using the interruptible PING method.
    ping -n %PING_COUNT% 127.0.0.1 >nul

    :: Check if the container is still running before checking its logs.
    docker ps -f "name=%TARGET_CONTAINER%" | findstr "%TARGET_CONTAINER%" >nul
    IF %ERRORLEVEL% NEQ 0 (
        echo [Background Task] ERROR: Container '%TARGET_CONTAINER%' stopped unexpectedly. Aborting monitor.
        exit /B
    )

    :: Check the container's logs for the "ready" message.
    docker logs %TARGET_CONTAINER% 2>&1 | findstr /C:"* Running on local URL:"
    IF %ERRORLEVEL% NEQ 0 (
        goto log_check_loop
    )

    echo [Background Task] SUCCESS! Application is ready.
    echo [Background Task] Opening URL in your browser: %TARGET_URL%
    start "" "%TARGET_URL%"
    exit /B