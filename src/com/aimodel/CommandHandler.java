package com.aimodel;

import java.util.Scanner;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class CommandHandler {
    public void handleCommand(String input) {
        System.out.println("\u001B[36mDEBUG: Received input: '" + input + "'\u001B[0m");
        if (input.equalsIgnoreCase("/help")) {
            showHelp();
        } else if (input.startsWith("/train ")) {
            String topic = input.substring(7).trim();
            if (topic.isEmpty()) {
                System.out.println("\u001B[31mPlease specify a topic to train on.\u001B[0m");
                return;
            }
            System.out.println("\u001B[33mStarting training on topic: \"" + topic + "\"...\u001B[0m");
            try {
                String output = PythonBridge.runPythonScript("scripts/brain.py", "train", topic);
                System.out.println(output);
                System.out.println("\u001B[32mTraining completed. I am ready to answer questions.\u001B[0m");
            } catch (Exception e) {
                System.out.println("\u001B[31mError during training: " + e.getMessage() + "\u001B[0m");
            }
        } else if (input.startsWith("/ask ")) {
            String question = input.substring(5).trim();
            if (question.isEmpty()) {
                System.out.println("\u001B[31mPlease ask a question.\u001B[0m");
                return;
            }
            System.out.println("\u001B[33mThinking about: \"" + question + "\"...\u001B[0m");
            try {
                String output = PythonBridge.runPythonScript("scripts/brain.py", "query", question);
                System.out.println(output);
            } catch (Exception e) {
                System.out.println("\u001B[31mError during query: " + e.getMessage() + "\u001B[0m");
            }
        } else if (input.equalsIgnoreCase("/list")) {
            System.out.println("\u001B[33mListing trained topics...\u001B[0m");
            try {
                String output = PythonBridge.runPythonScript("scripts/brain.py", "list");
                System.out.println(output);
            } catch (Exception e) {
                System.out.println("\u001B[31mError listing topics: " + e.getMessage() + "\u001B[0m");
            }
        } else if (input.equalsIgnoreCase("/info")) {
            System.out.println("\u001B[33mGetting system info...\u001B[0m");
            try {
                String output = PythonBridge.runPythonScript("scripts/brain.py", "info");
                System.out.println(output);
            } catch (Exception e) {
                System.out.println("\u001B[31mError getting info: " + e.getMessage() + "\u001B[0m");
            }
        } else if (input.equalsIgnoreCase("/clear")) {
            System.out.println("\u001B[33mClearing database...\u001B[0m");
            try {
                String output = PythonBridge.runPythonScript("scripts/brain.py", "clear");
                System.out.println(output);
            } catch (Exception e) {
                System.out.println("\u001B[31mError clearing database: " + e.getMessage() + "\u001B[0m");
            }
        } else if (input.equalsIgnoreCase("/ready")) {
            System.out.println("\u001B[33mChecking readiness...\u001B[0m");
            try {
                String output = PythonBridge.runPythonScript("scripts/brain.py", "info");
                // Parse output to see if we have any documents
                if (output.contains("Documents (topics): 0")) {
                    System.out.println("I have not been trained yet. Please train me first using /train <topic>.");
                } else {
                    System.out.println("I am ready! I have been trained and can answer questions.");
                }
            } catch (Exception e) {
                System.out.println("\u001B[31mError checking readiness: " + e.getMessage() + "\u001B[0m");
            }
        } else if (input.equalsIgnoreCase("/exit")) {
            System.out.println("\u001B[32mGoodbye!\u001B[0m");
            System.exit(0);
        } else {
            System.out.println("\u001B[31mUnknown command. Type /help for available commands.\u001B[0m");
        }
    }

    private void showHelp() {
        System.out.println("\u001B[36mAvailable commands:\u001B[0m");
        System.out.println("  /help    - Show this help message");
        System.out.println("  /train <topic> - Train the model on a topic");
        System.out.println("  /ask <question> - Ask a question to the model");
        System.out.println("  /list    - List trained topics");
        System.out.println("  /info    - Show system information");
        System.out.println("  /clear   - Clear the database");
        System.out.println("  /ready   - Check if the system is ready");
        System.out.println("  /exit    - Exit the program");
        System.out.println();
    }
}
