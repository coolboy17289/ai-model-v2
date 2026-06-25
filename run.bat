@echo off
REM Compile Java source files
javac -d bin src\com\aimodel\*.java
REM Check if compilation succeeded
if errorlevel 1 (
    echo Compilation failed.
    exit /b 1
)
REM Run the Main class
java -cp bin com.aimodel.Main
