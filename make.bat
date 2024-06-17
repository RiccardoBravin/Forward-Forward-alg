@echo off
cls
echo Compiling...
g++ .\*.cpp -std=c++17 -O3 -fopenmp -o .\run.exe

if %errorlevel% neq 0 (
    echo Compilation failed.
    exit /b %errorlevel%
)

echo Compilation successful. Running...
.\run.exe