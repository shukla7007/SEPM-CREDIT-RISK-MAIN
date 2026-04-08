
import random

# Generate a random number between 1 and 100
n = random.randint(1, 100)
a = -1  # Initial value for the guess
guesses = 0  # Counter for the number of attempts

# Loop until the user guesses the correct number
while a != n:
    guesses += 1  # Increment the guess counter
    a = int(input("Guess the number: "))

    if a > n:
        print("Lower number please")
    elif a < n:
        print("Higher number please")

# Once guessed correctly, display the number of attempts
print(f"You have guessed the number correctly in {guesses} attempts!")



































# import random

# def guess_number_game():
#     # Generate a random number between 1 and 100
#     number_to_guess = random.randint(1, 100)
#     guess = None
#     attempts = 0

#     print("Welcome to the Number Guessing Game!")
#     print("I have selected a number between 1 and 100. Can you guess it?")

#     # Loop until the user guesses the correct number
#     while guess != number_to_guess:
#         try:
#             # Get user input and convert it to an integer
#             guess = int(input("Enter your guess: "))
#             attempts += 1

#             # Check if the guess is too high or too low
#             if guess > number_to_guess:
#                 print("Lower number please!")
#             elif guess < number_to_guess:
#                 print("Higher number please!")
#             else:
#                 print(f"Congratulations! You guessed the correct number in {attempts} attempts.")
#         except ValueError:
#             print("Please enter a valid integer.")

# # Run the game
# guess_number_game()
# 