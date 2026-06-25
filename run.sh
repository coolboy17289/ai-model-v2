#!/bin/bash
# Compile Java source files
javac -d bin src/com/aimodel/*.java
# Check if compilation succeeded
if [ $? -ne 0 ]; then
    echo 'Compilation failed.'
    exit 1
fi
# Run the Main class
java -cp bin com.aimodel.Main
