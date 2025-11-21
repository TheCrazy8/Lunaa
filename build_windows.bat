@echo off
echo Building Lunaa for Windows...
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Build the application
echo Building executable with PyInstaller...
pyinstaller lunaa.spec

echo.
echo Build complete! Check the 'dist' folder for the executable.
echo.
pause
