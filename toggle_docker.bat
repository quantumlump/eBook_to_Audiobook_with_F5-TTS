@echo off
setlocal enabledelayedexpansion

pushd "C:\Program Files\Docker\Docker\resources\bin"

set ACTION_TAKEN=0

:: Part 1: Find any PAUSED containers and UNPAUSE them
for /f "tokens=*" %%i in ('docker.exe ps -q --filter "status=paused"') do (
    docker.exe unpause %%i
    set ACTION_TAKEN=1
)

:: If we took action, notify the user and exit
if !ACTION_TAKEN! == 1 (
    popd
    wscript.exe "C:\Users\Sarah\Desktop\Browser_Downloads\eBook_to_Audiobook_F5-TTS\Backup F5-TTS_eBook_to_Audiobook_53\notify.vbs" "Docker Containers Resumed"
    goto :eof
)

:: Part 2: If we did NOT unpause, find RUNNING containers and PAUSE them
for /f "tokens=*" %%j in ('docker.exe ps -q --filter "status=running"') do (
    docker.exe pause %%j
    set ACTION_TAKEN=1
)

:: If we took action, notify the user
if !ACTION_TAKEN! == 1 (
    popd
    wscript.exe "C:\Users\Sarah\Desktop\Browser_Downloads\eBook_to_Audiobook_F5-TTS\Backup F5-TTS_eBook_to_Audiobook_53\notify.vbs" "Docker Containers Paused"
)

popd
endlocal