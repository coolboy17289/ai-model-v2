package com.aimodel;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // Display banner
        System.out.println("\u001B[30m\u001B[46m"); // Black text, cyan background
        System.out.println("███╗   ██╗ ██████╗ ██████╗ ███████╗ ██████╗ ██╗   ██╗██╗   ██╗");
        System.out.println("████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔═══██╗██║   ██║╚██╗ ██╔╝");
        System.out.println("██╔██╗ ██║██║   ██║██████╔╝█████╗  ██║   ██║██║   ██║ ╚████╔╝ ");
        System.out.println("██║╚██╗██║██║   ██║██╔══██╗██╔══╝  ██║   ██║██║   ██║  ╚██╔╝ ");
        System.out.println("██║ ╚████║╚██████╔╝██║  ██║███████╗╚██████╔╝╚██████╔╝   ██║  ");
        System.out.println("╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝    ╚═╝  ");
        System.out.println("\u001B[0m"); // Reset colors
        System.out.println("\u001B[32mAI Model Training & Querying System\u001B[0m");
        System.out.println("Type /help for available commands.\n");

        Scanner scanner = new Scanner(System.in);
        CommandHandler commandHandler = new CommandHandler();

        System.out.print("\u001B[32mai-model > \u001B[0m"); // Green prompt
        while (scanner.hasNextLine()) {
            String input = scanner.nextLine().trim();
            if (input.isEmpty()) {
                System.out.print("\u001B[32mai-model > \u001B[0m");
                continue;
            }
            commandHandler.handleCommand(input);
            System.out.print("\u001B[32mai-model > \u001B[0m");
        }
        scanner.close();
    }
}
