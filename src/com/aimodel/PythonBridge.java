package com.aimodel;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class PythonBridge {
    /**
     * Runs a Python script with the given arguments and displays a spinner while it runs.
     * @param scriptPath The path to the Python script.
     * @param args The arguments to pass to the script.
     * @return The output of the script as a string.
     * @throws IOException If an I/O error occurs.
     * @throws InterruptedException If the current thread is interrupted while waiting.
     */
    public static String runPythonScript(String scriptPath, String... args) throws IOException, InterruptedException {
        // Build the command: python script.py arg1 arg2 ...
        ProcessBuilder pb = new ProcessBuilder();
        pb.command().add("python");
        pb.command().add(scriptPath);
        for (String arg : args) {
            pb.command().add(arg);
        }
        pb.redirectErrorStream(true); // Merge error stream with input stream
        Process process = pb.start();

        // Start a thread to display a spinner while the process runs
        Thread spinnerThread = new Thread(() -> {
            String[] spinner = {"|", "/", "-", "\\"};
            int i = 0;
            try {
                while (process.isAlive()) {
                    System.out.print("\r\u001B[33mProcessing " + spinner[i % 4] + "\u001B[0m");
                    Thread.sleep(100);
                    i++;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            System.out.print("\r"); // Clear the line after done
        });
        spinnerThread.setDaemon(true);
        spinnerThread.start();

        // Read the output
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append(System.lineSeparator());
            }
        }

        int exitCode = process.waitFor();
        spinnerThread.interrupt(); // Stop the spinner thread

        if (exitCode != 0) {
            throw new IOException("Python script exited with code " + exitCode);
        }

        return output.toString().trim();
    }
}
