# import random

# random_integer = random.randint(1,10)
# print(random_integer) to generate a random number 

# import my_module
# print(my_module.pi) to import a code from one file to other 

# import random
# random_float = random.random()
# print(random_float) #to generate a random float Number

# import random 
# random_decimal = random.randint(0,5)/100
# print(random_decimal) to create a decimal number between 

#or we can use a diffrent method to generate decimal number 
#as a random float number is anywhere between 0.00000000-0.99999999
 
# import random

# random_float = random.random()*5
# print(random_float)

# import random 

# random_side = random.randint(0,1)
# if random_side == 1:
#   print("Heads")
# else :
#   print("Tails")

# list data structure always open with square brackets 
# list = [1,2,3,4,5,6,7,8,9]

# to extract an element from list and to print it 

# dirty_dozen = ["Strawberries", "Spinach", "Kale", "Nectarines", "Apples", "Grapes", "Peaches", "Cherries", "Pears", "Tomatoes", "Celery", "Potatoes"]
# print(dirty_dozen[0])
# to change items in  the list 
# dirty_dozen[0] = "Avocado"
# print(dirty_dozen[0])
# to add items in the list
# dirty_dozen.append("Avocado") we use aapend to add item in the list 
# print(dirty_dozen)
#append adds 1 item in the list to add bunch of the item in the list we use extend 
# dirty_dozen.extend(["Avocado","Banana"])

# import random 
# names_string = "John, Alice, Bob, Charlie"
# names = names_string.split(", ") #would split the string into a list of names:names = ["John", "Alice", "Bob", "Charlie"]
# num_items = len(names)
# random_choice = random.randint(0,num_items-1)#as it is starting from 0
# print(names[random_choice]+" is going to buy the meal today!")

#nested list list within a list 
# fruits = ["Strawberries","nectarine","apples","oranges"]
# vegetables = ["Spinach", "Kale", "Tomatoes","Potatoes"]
# dirty_dozen = [fruits,vegetables]
# print(dirty_dozen)
# output=[['Strawberries', 'nectarine', 'apples', 'oranges'], ['Spinach', 'Kale', 'Tomatoes', 'Potatoes']]

#to access the list within the list
# print(dirty_dozen[0][1]) #output-nectarine
# print(dirty_dozen[1][2]) #output-Tomatoes
# Here's the breakdown:

# dirty_dozen is a list that contains two inner lists: fruits and vegetables.
# dirty_dozen[0] accesses the first inner list, which is fruits.
# dirty_dozen[0][1] accesses the second element of the fruits list, which is "nectarine".

# line1 = ["⬜️","️⬜️","️⬜️"]
# line2 = ["⬜️","⬜️","️⬜️"]
# line3 = ["⬜️️","⬜️️","⬜️️"]
# map = [line1, line2, line3]
# print("Hiding your treasure! X marks the spot.")
# position = input() # Where do you want to put the treasure?
# # 🚨 Don't change the code above 👆
# # Write your code below this row 👇
# letter = position[0].lower()
# abc = ["a", "b", "c"] #you can choose wheteher you want to use upper case or lower case 
# letter_index =abc.index(letter)
# #if the user inputs "a1", then letter would be "a", and abc.index(letter) would return 0, because "a" is the first element in the abc list. If the user inputs "b1", then letter would be "b", and abc.index(letter) would return 1, because "b" is the second element in the abc list.
# #abc.index(letter) is a method call that searches for the letter in the abc list and returns its index (i.e. its position in the list)
# number_index = int(position[1]) - 1#When the user inputs a position, they are likely to enter a number starting from 1, not 0. For example, if they want to place the treasure in the first row, they would enter "a1", not "a0".

# #By subtracting 1 from the input number, we convert the user's input to a valid index for the map list. This ensures that the treasure is placed at the correct position in the list

# map[number_index][letter_index] = "X"
# print(f"{line1}\n{line2}\n{line3}")
# import random
# rock = '''
#     _______
# ---'   ____)
#       (_____)
#       (_____)
#       (____)
# ---.__(___)
# '''

# paper = '''
#     _______
# ---'   ____)____
#           ______)
#           _______)
#          _______)
# ---.__________)
# '''

# scissors = '''
#     _______
# ---'   ____)____
#           ______)
#        __________)
#       (____)
# ---.__(___)
# '''
# game_images= [rock,paper,scissors]#we created a list a list carries in an prdered manner 
# #Write your code below this line 👇
# user_choice = int(input("What do you chosse?  Type 0 for Rock, 1 for Paper or 2 for Scissors.\n"))#user_choice is a variable that stores the user's choice
# print(game_images[user_choice])#By using user_choice as an index to access the game_images list, the corresponding image is printed to the console. This allows the user to see a visual representation of their choice.
# computer_choice = random.randint(0,2)
# #computer_choice is a variable that stores the computer's choice
# print(f"Computer chose:")
# print(game_images[computer_choice])


# if user_choice == 0 and computer_choice == 2:
#     print("You win!")
# elif user_choice == 2 and computer_choice == 0:
#      print("You lose!")
# elif user_choice == 1 and computer_choice == 0:
#      print("You win!")
# elif user_choice == computer_choice:
#      print("It's a draw!")
# elif user_choice == 0 and computer_choice == 1:
#      print("You lose!")
# elif user_choice == 1 and computer_choice == 2:
#      print("You win!")
# elif user_choice == 2 and computer_choice == 1:
#      print("You lose!")
# else:
#      print("You typed an invalid number, you lose!")


# day 5 
# for loops 

# fruits = ["Apple","Peach","Pear"]
# for fruit in fruits:
#     print(fruit)
#     print(fruit + " Pie") 
#     print(fruits) #it is going to print 3 times as it is inside the for loop 
# print(fruits)#it is going to be printed only one time as it is outside loop 

# Input a Python list of student heights
# student_heights = input().split()#is used to get a list of student heights from the user.
#The reason for using split() is to allow the user to enter multiple values separated by spaces, and then convert those values into a list that can be easily processed by the program.
#Without split(), the program would not be able to handle multiple input values separated by spaces. It would treat the entire input string as a single value, which is not what we want.

# for n in range(0, len(student_heights)):
#   student_heights[n] = int(student_heights[n])
# is used to iterate over the list and convert each string value to an integer using student_heights[n] = int(student_heights[n]). This is necessary because the split() method returns a list of strings, but the program needs to work with integer values.
# 🚨 Don't change the code above 👆
  
# Write your code below this row 👇
# total_height = 0
# for height in student_heights:
#   total_height+= height
# print(f"total height = {total_height}")

# number_of_students = 0
# for student in student_heights:
#   number_of_students += 1
# print(f"number of students = {number_of_students}")

# average_height = round(total_height/number_of_students)
# print(f"average height = {average_height}")

#to find the highest score in class 
# Input a list of student scores
# student_scores = input().split()
# for n in range(0, len(student_scores)):
#   student_scores[n] = int(student_scores[n])

# # Write your code below this row 👇
# highest_score = 0
# for score in student_scores:
#   if score > highest_score:
#     highest_score = score
#     #print(highest_score)
# print(f"The highest score in the class is: {highest_score}")

#using for loops with range function
# for number in range(1,11):#it will print from 1 to 10 if we would take (1,10) then only till 9 would get print 
#   print(number)

# total = 0
# for number in range(1,101):
#   total += number #TO FIND SUM OF NUMBERS FROM 1 TO 100
#   print(total)
# print(f"total = {total}")

#for even number 
# target = int(input()) # Enter a number between 0 and 1000
# even_sum = 0
# for number in range(1, target+1):
#   if number%2 == 0:
#     even_sum += number
# print(even_sum)

# The Fizz Buzz game is a classic programming exercise where you need to print out numbers from 1 to 100, but with a twist:

# If the number is a multiple of 3, you print "Fizz" instead of the number.
# If the number is a multiple of 5, you print "Buzz" instead of the number.
# If the number is a multiple of both 3 and 5, you print "FizzBuzz" instead of the number.
# target =100
# for number in range(1,target+1):
#  if number%3 == 0 and number%5 == 0:
#   print("FizzBuzz")
#  elif number%3==0:
#   print("Fizz")
#  elif number%5==0:
#   print("Buzz")
#  else:
#   print(number)

#Password Generator Project
import random
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

print("Welcome to the PyPassword Generator!")
nr_letters= int(input("How many letters would you like in your password?\n")) 
nr_symbols = int(input(f"How many symbols would you like?\n"))
nr_numbers = int(input(f"How many numbers would you like?\n"))

#Eazy Level - Order not randomised:
#e.g. 4 letter, 2 symbol, 2 number = JduE&!91
# password = ""
# #nr_letters = 4 lets take 4 letter would be written by the user 
# for char in range(1,nr_letters+1):
# #1-4
#    random_char = random.choice(letters)
#    password += random_char
#    #you can directly use password += random.choice(letters)
#    #for symbols and number we need to dom the same 
#    for char in range(1,nr_symbols+1):
#       #1-4
#       random_char = random.choice(symbols)
#       password += random_char
#       #you can directly use password += random.choice(letters)
#       for char in range(1,nr_numbers+1):
#          #1-4
#          random_char = random.choice(numbers)
#          password += random_char
#     print(password)

password_list = []# we are storing the password in list to shuffel it so that it does not have any pattern

for char in range(1, nr_letters + 1):
  password_list.append(random.choice(letters))#both += and append does the same thing 
#   It selects a random character from the letters string using random.choice().
# It adds this random character to the end of the password_list using append().
# It repeats this process for the number of letters specified by the user.


for char in range(1, nr_symbols + 1):
  password_list += random.choice(symbols)

for char in range(1, nr_numbers + 1):
  password_list += random.choice(numbers)

print(password_list)
random.shuffle(password_list)#using shuffel to redorder the list 
print(password_list)
# to convert the list into a string we create an empty string and using for loop 
# password = ""
# for char in password_list:
#   password += char

#or we can use join method instead of concatenating string together   
password = []
password_string = ''.join(password_list)
print(password_string)  # Output: 'abcde'

print(f"Your password is: {password_string}")



#Hard Level - Order of characters randomised:
#e.g. 4 letter, 2 symbol, 2 number = g^2jk8&P
  
