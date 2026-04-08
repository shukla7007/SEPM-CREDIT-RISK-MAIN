# friends = ["alex", "eren","edwin", "aron"]
# for friend in friends:
#  print("hello "+friend)


# for i in range(10):
#  print("Hello world")

# name = "anshul"
# print("Hello"+ name)

# print(5*10)

# print(type(2))
# print(type("a"))

# base = 6
# height = 4
# area = (base*height)/2
# print("the area of traingle is:"+str(area))#as area is an integer to concatenate we need to convert it into string

# The following lines assign the variable to the left of the = 
# assignment operator with the values and arithmetic expressions 
# on the right side of the = assignment operator.
# hotel_room = 100
# tax = hotel_room * 0.08
# total = hotel_room + tax
# room_guests = 4
# share_per_person = total/room_guests


# # This line outputs the result of the final calculation stored
# # in the variable "share_per_person"
# print("Each person needs to pay: " + str(share_per_person)) # change a data type

# # The following 5 lines assign strings to a list of variables.
# salutation = "Dr."
# first_name = "Prisha"
# middle_name = "Jai"
# last_name = "Agarwal"
# suffix = "Ph.D."
 
# print(salutation + " " + first_name + " " + middle_name + " " + last_name + ", " + suffix) 
# # The comma as a string ", " adds the conventional use of a comma plus a 
# # space to separate the last name from the suffix.
 
# # Alternatively, you could use commas in place of the + connector:
# print(salutation, first_name, middle_name, last_name,",", suffix)
# # However, you will find that this produces a space before a comma within a string.


# # The following code causes a type error between a string 
# # and an integer:

# print("5 * 3 = " + (5*3)) 


# # Resolution: 
# # print("5 * 3 = " + str(5*3))
# #
# # To avoid a type error between the string and the integer within the
# # print() function, you can make an explicit data type conversion by
# # using the str() function to convert the integer to a string.


# numerator = 7
# denominator = 0   # Possible resolution: Change the denominator value 
# result = numerator / denominator
# print(result)


# # One possible assumption for a number divided by zero error might
# # include the issue of a null value as a denominator (could happen when
# # using a loop to iterate over values in a database). In such cases, the
# # desired outcome may be to leave the numerator value intact. The
# # numerator value can be preserved by reassigning the denominator with 
# # the integer value of 1. The result would then equal the numerator.


# def greeting(name, department):
#     print("Welcome, " + name)
#     print("You are part of " + department)
    
# greeting("Blake", "Software engineering")
# greeting("Ellis", "Software engineering")
# output
# Welcome, Blake
# You are part of Software engineering
# Welcome, Ellis
# You are part of Software engineering


# def person(name,age):#to define a function we use def keyword greeting is function name ,name is parameter of function and is between ()
#     print("hello "+ name + " your age is:"+str(age))
# person("Blake", 13)
# person("Ellis", 12)


# time_list = [12,13,14,15,16,17,18,19]
# print(sorted(time_list))

# def area_triangle(base,height):
#     return base*height/2
# area_a = area_triangle(5,4)
# area_b = area_triangle(7,3)
# sum = area_a + area_b
# print(sum)

# def convert_seconds(seconds):
#     hours = seconds // 3600
#     minutes = (seconds - hours * 3600) // 60
#     remaining_seconds = seconds - hours * 3600 - minutes * 60
#     return hours, minutes, remaining_seconds
 
# hours, minutes, seconds = convert_seconds(5000)# the 5000 is essentially the input to the function, and the function will process this input to produce the desired output, 
# print(hours, minutes, seconds)

# name = "anshul"
# number =len(name)*3
# print("hello "+name + " your luck number is:"+str(number))

# for multiple name we can use def 

# def lucky_number(name):
#     number = len(name)*3
#     print("hello "+name + " your luck number is:"+str(number))
# lucky_number("anshul")#always remember the function name defined later should be intended with function defined in the starting 
# lucky_number("sachin")
# lucky_number("ellis")

# def circle_area(radius):
#     pi=3.14
#     area=pi*radius*radius
#     print(area)
# circle_area(5)

# This function calculates the number of days in a variable number of 
# years, months, and days. These variables are provided by the user and
# are passed to the function through the function’s parameters.
# def find_total_days(years, months, days):
# # Assign a variable to hold the calculations for the number of days in
# # a year (years*365) plus the number of days in a month (months*30) plus
# # the number of days provided through the "days" parameter variable.
#     my_days = (years*365) + (months*30) + days
# # Use the "return" keyword to send the result of the "my_days"  
# # calculation to the function call. 
#     return my_days#If the function directly printed the values, the caller would have no control over how the values are used.
#  #By returning values, the function can be kept simple and focused on its main task. The printing can be handled separately, which makes the code more organized and easier to read.
# # Function call with user provided parameter values. 
# print(find_total_days(2,5,23))


#print(10>1)#output true
# print(1<"1")#output type error 
#print("yellow">"red" and "brown">"green")#false 
# print("yellow">"red" or "brown">"green")#true
# The expression print("yellow">"red" or "brown">"green") is indeed True, but it's not because of the comparison between the strings.

# In Python, when you compare two strings using the > operator, it performs a lexicographic comparison, which means it compares the strings character by character based on their ASCII values.

# In this case, "yellow" is indeed greater than "red" because the ASCII value of 'y' (121) is greater than the ASCII value of 'r' (114).

# However, the expression "brown">"green" is actually False, because the ASCII value of 'b' (98) is less than the ASCII value of 'g' (103).


# print((6*3 >= 18) and (9+9 <= 36/2)) true 
# print("Nairobi" < "Milan" and "Nairobi" > "Hanoi") just check the first letter following the other letters and see which is greater as which will be greater will have greter ascii code 


# x = 2*4 > 5
# print("The value of x is:")
# print(x)# true 

# print("") #for blank line 
# print("the inverse value of x is:")
# print(not x) #it will print the opposite value of answer #false

# def hint_username(username):
#     if len(username)<3:
#         print("Invalid username. Must be atleast 3 characters long.")

# def is_positive(number):
#     if number > 0:
#      return True #we are not using print as we are making boolean expression 
#     else:
#        return False

# def hint_username(username):
#     if len(username) < 3:
#         print("Invalid username. Must be at least 3 characters long")
#     elif len(username) > 15:
#         print("Invalid username. Must be at most 15 characters long")
#     else:
#         print("Valid username")

# def calculate_storage(filesize):
#     block_size = 4096
#     # Use floor division to calculate how many blocks are fully occupied
#     full_blocks = filesize // 4096
#     # Use the modulo operator to check whether there's any remainder
#     partial_block_remainder = filesize % block_size
#     # Depending on whether there's a remainder or not, return
#     # the total number of bytes required to allocate enough blocks
#     # to store your data.
#     if partial_block_remainder > 0:
#         return (full_blocks + 1) * block_size
#     return full_blocks * block_size

# print(calculate_storage(1))    # Should be 4096
# print(calculate_storage(4096)) # Should be 4096
# print(calculate_storage(4097)) # Should be 8192
# print(calculate_storage(6000)) # Should be 8192


# # This function rounds a variable number up to the nearest 10x value
# def round_up(number):
#   x = 10
# # The floor division operator will calculate the integer value of
# # "number" divided by x: 35 // 10 will return the integer 3.
#   whole_number = number // x
# # The modulo operator will calculate the remainder value of "number"
# # divided by x: 35 % 10 will return the remainder value 5.
#   remainder = number % x
# # If the remainder is greater than or equal to 5: 
#   if remainder >= 5: 
# # Return x multiplied by the (whole_number+1) to round up
#     return x*(whole_number+1)
# # Else, return x multiplied by the whole_number to round down
#   return x*whole_number
 
# # Calls the function with the parameter value of 35.
# print(round_up(35)) # Should print 40

# def product(a,b):
#    return(a*b)
# print(product(2,3))

# def difference(a,b):
#     return(a-b)
# def sum(a,b):
#     return(a+b)

# print(difference(sum(2,3), sum(3,3)))#output -1

# def get_remainder(x, y):
 
#   if x == 0 or y == 0 or x ==y:
#     remainder = 0
#   else:
#     remainder = (x % y) / y
#   return remainder


# print(get_remainder(10, 3))

# x =0 #giving inital value to the variable 
# while x<5:
#     print("Not there yet,x="+str(x))
#     x = x +1
# print("x="+str(x))
           

# def attempts(number):
#     x=1
#     while x<=number:
#         print("Attempt"+str(x))
#         x=x+1
#     print("done")

# attempts(5)

# while my_variable < 10:
#     print("Hello")
#     my_variable += 1
#This code will give a NameError
#Variable is not defined 


# my_variable = 5 here the variable is definefd first 
# while my_variable < 10:
#     print("hello")
#     my_variable +=1


# x =1 
# sum = 0
# while x<=10:
#     sum = sum + x
#     x= x +1 
    

# product = 1
# while x<=10:
#     product = product*x
#     x=x+1

# print(sum,product)


# In this code, there's an initialization problem that's causing our function to behave incorrectly. Can you find the problem and fix it?
# def count_down(start_number):
#   while (current > 0):
#     print(current)
#     current -= 1
#   print("Zero!")

#correct code 
# def count_down(start_number):
#   current = start_number
#   while (current > 0):
#     print(current)
#     current -= 1
#   print("Zero!")

# multiplier = 1
# result = multiplier * 5#to check if the inital value result does not cross 50 as while loop would run only if the condition were true 
# #we need to define the result first before running the while loop
# while result <= 50:
#     print(result)
#     multiplier += 1
#     result = multiplier * 5
    #we can't write print after above 2 line as if we write print(result)all numbers would be printed except 5 to include 5 wee need to print before iterating after iterating others number would be printed 
# print("Done")

#If the line result = multiplier * 5 were not written twice, the value of result would not be updated correctly inside the loop, and the loop would not terminate correctly


# def addition_table(given_number):
#     iterated_number = 1
  

#     while iterated_number <= 5:
#           my_sum = given_number + iterated_number
#           if my_sum > 20:
#            break
#           print(str(given_number), "+", str(iterated_number), "=", str(my_sum))
#           iterated_number += 1

# addition_table(5)
# addition_table(17)
# addition_table(30)

# for x in range(5):
#     print(x)
#output 
# 0
# 1
# 2
# 3
# 4


# friends =["anshul","ayush","arav"]
# for friend in friends:
#     print("hi "+friend)

# values = [23,42,54,26,27,28]
# sum = 0
# length = 0
# for value in values:
#     sum = sum+value
#     length = length+1
#     average=sum/length

# print("Total sum: " + str(sum) + "-Average:" + str(average))


# product =1 
# for n in range(1,10):
#     product = product * n

# print(product)


# for n in range(1, 5, 6):  
#     print(n)
    #output 1
#The sequence would be: 1, 7, 13,... (but it would never reach 5, because the stop value is exclusive)


# for number in range(2,7+1):
#     print(number*3)# The loop should print 6, 9, 12, 15, 18, 21

# for left in range(7):
#   for right in range(left, 7):
#     print("[" + str(left) + "|" + str(right) + "]", end=" ")
#   print()


# teams = ['Dragons', 'wolves','pandas','unicorn']
# for home_team in teams:
#     for away_team in teams:
#       if home_team  != away_team:
#        print(home_team + " vs " + away_team)
# #in output it will create a combination of one team vs another till all team has been competed with each other 

# greeting = "hello"
# for char in greeting:
#   print(char)
# #h e l l o

# greeting = "hello"
# for i in range(len(greeting)):
# 	print(i)
#output 0
# 1
# 2
# 3
# 4



#while loop with indexing
# greeting = "hello"
# index = 0
# while index<len(greeting):
#     print(greeting[index])
#     index = index + 1
    #output h e l l o

#while loop with slicing:Using a while loop with slicing accomplishes the same thing that a while loop with indexing does
#it is just another way to write the while loop 
# greeting = "hello"

# numbers = [1, 2, 3, 4, 5]
# squared_numbers = [x ** 2 for x in numbers]
# print(squared_numbers)

#output [1, 4, 9, 16, 25]

#string slicing 
# string1 = "Greetings, Earthlings"
# print(string1[0])   # Prints “G”
# print(string1[4:8]) # Prints “ting”
# print(string1[11:]) # Prints “Earthlings”
# print(string1[:5])  # Prints “Greet”

# print(string1[-10:])     # Prints “Earthlings” again


# greetings = ["Hello", "world"]
# print(" ".join(greetings))  # Prints "Hello world"
# You can also concatenate a combination of strings and variables like in the following example.
# name = "Alice"
# print("Hello, " + name + "!")  # Prints "Hello, Alice!"


#An optional way to slice an index is by the stride argument, indicated by using a double colon.
# print(string1[0::2])    # Prints “Getns atlns”
# print(string1[::-1])    # Prints “sgnilhtraE ,sgniteerG”

# def format_phone(phonenum):
#     area_code = "(" + phonenum[:3] + ")"
#     exchange = phonenum[3:6]
#     line = phonenum[-4:]
#     return area_code + " " + exchange + "-" + line

# for x in 25:
#     print(x)

# #this will produce an error

# for x in range(25):
#     print(x)

# #this will make the error go away

# def greet_friends(friends):
#     for friend in friends:
#         print("Hi " + friend)

# greet_friends(['Taylor', 'Luisa', 'Jamaal', 'Eli'])

# #output
# Hi Taylor
# Hi Luisa
# Hi Jamaal
# Hi El

# def greet_friends(friends):
#     for friend in friends:
#         print("Hi " + friend)

# greet_friends("Barry")

#Hi B
# Hi a
# Hi r
# Hi r
# Hi y

#Example of nested for loops:
# This code demonstrates the outer and inner loop iterations of a pair 
# of nested for loops. Click "Run" to see the results. The outer loop
# will run twice for the range pointer positions [0, 1] in range(2).
# The inner loop will run 4 times for the range pointer positions 
# [0, 1, 2, 3] in range(3+1) or range(4) each time the outer loop runs.
# So, the inner loop will execute 8 times in total.

# for x in range(2):
#     print("This is the outer loop iteration number " + str(x))
#     for y in range(3+1):
#         print("Inner loop iteration number " + str(y))
#     print("Exit inner loop")

# This is the outer loop iteration number 0
# Inner loop iteration number 0
# Inner loop iteration number 1
# Inner loop iteration number 2
# Inner loop iteration number 3
# Exit inner loop
# This is the outer loop iteration number 1
# Inner loop iteration number 0
# Inner loop iteration number 1
# Inner loop iteration number 2
# Inner loop iteration number 3
# Exit inner loop


# for x in range(7):
#     if x % 2 == 0:
#         print(x)

# # The loop should print 0, 2, 4, 6

# # As a list comprehension:
# even_numbers = [x for x in range(7) if x % 2 == 0]
# print(even_numbers)
#output 0
# 2
# 4
# 6
# [0, 2, 4, 6]

#the function “digits(n)” to count how many digits the given number has.
# def digits(n):
#     count = 0 count = 0: #Initializes a variable count to 0, which will be used to store the number of digits in the input number n
#     if n == 0:#if n == 0: count += 1: This is a special case to handle the input n = 0. Since 0 has one digit, we increment the count variable to 1
#       count += 1
#     while n!=0: # Complete the while loop condition #while n!= 0:: This loop will continue until n becomes 0.

#         # Complete the body of the while loop. This should include 
#         # performing a calculation and incrementing a variable in the
#         # appropriate order.  
#         n//=10 
#         count+= 1
#     return count

#n/10 performs true division, which means it returns a float result. This is the default behavior for division in Python 3.x.

#n//10 (Floor Division)n//10 performs floor division, which means it returns an integer result, rounded down to the nearest whole number. This is also known as integer division.

# for x in range(10):#it will go from  0 to 9
#     for y in range(x):#for range(9) it will print till 8
#         print(y)

# "example" * 3 #output exampleexampleexample

# name = "anshul"
# print(name[1])
#n

# fruit = "Pineapple"
# print(fruit[:4])#Pine
# print(fruit[4:])#apple


# message = "A kong string with a silly typo"
# new_message = message[0:2] + "l" + message[3:]
# print(new_message)
#output A long string with a silly typo

# pets="Cats & Dogs"
# pets.index("&")#4
# pets.index("C")#0
# pets.index("Dog")#6 when dog first occur 
# pets.index("s")#3 as first occurence of character s
# print(pets.index("s"))
#output is 3 

# " yes ".strip()#this function is used to get rid of spaces in a string

#Formatting strings
# name = "anshul"
# number = len(name)*3
# print("hello {},your luck number is {}".format(name,number))
#output hello anshul,your luck number is 18

# name = "Manny"
# print("Your lucky number is {number}, {name}.".format(name=name, number=len(name)*3))
# #output Your lucky number is 18, Manny.

# price = 7.5
# with_tax = price * 1.09
# print(price, with_tax)
# print("Base price: ${:.2f}. With Tax: ${:.2f}".format(price, with_tax))
#Here's a breakdown of what :.2f does:

# . is the decimal point separator.
# 2 is the precision specifier, which means the number of digits to display after the decimal point.
# f is the format code for a fixed-point number (i.e., a floating-point number with a fixed number of decimal places).
#When you use :.2f in a format string, it tells Python to format the corresponding value (in this case, price and with_tax) as a floating-point number with two digits after the decimal point.

#program to convert fahereneit into celcius 
# def to_celsius(x):
#     return (x-32)*5/9

# for x in range(0,101,10):#for x in range(0, 101, 10):#This is a for loop that iterates over a range of numbers from 0 to 100, incrementing by 10 each time (i.e., 0, 10, 20,..., 100).
#   print("{:>3} F | {:>6.2f} C".format(x, to_celsius(x)))

  #{:>3}: This is a format specifier for the first argument x. The > symbol means "right-align" the value, and the 3 specifies the minimum field width. This means that the value of x will be printed in a field that is at least 3 characters wide, right-aligned.
  #:>6.2f}: This is a format specifier for the second argument to_celsius(x). The > symbol means "right-align" the value, the 6 specifies the minimum field width, and the .2f specifies that the value should be printed as a floating-point number with two digits after the decimal point.
  #So, when you put it all together, the format string {:>3} F | {:>6.2f} C will print the value of x (in Fahrenheit) right-aligned in a field of at least 3 characters, followed by the string " F | ", followed by the value of to_celsius(x) (in Celsius) right-aligned in a field of at least 6 characters, with two digits after the decimal point, followed by the string " C".\

# for c in "abcde":
#         print(c) #The loop for c in "abcde": is iterating over each character in the string "abcde". The variable c takes on the value of each character in the string, one at a time, during each iteration of the loop.

#print("abc" in "abcde")     # prints True
#print("abcde"[2])           # prints "c"
#print("abcde"[0:2]) # prints "ab"

#print("AaBbCcDdEe".lower())             # prints "aabbccddee"

#print("   Hello   ".lstrip())           # prints "Hello   "

#print("Hello   ".rstrip())               # prints "Hello"


# test = "How much wood would a woodchuck chuck"
# print(test.count("wood"))  # prints 2

# print("12345".isnumeric()) # prints True
# print("-123.45".isnumeric()) # prints False

#string.isalpha() - Returns True if there are only letters in the string. If not, returns False.
# print("xyzzy".isalpha())
# prints True

#test = "How-much-wood-would-a-woodchuck-chuck"
#print(test.split("-")) # prints ['How', 'much', 'wood', 'would', 'a', 'woodchuck', 'chuck']
#print(test.replace("wood", "plastic"))  # prints "How much plastic would a plasticchuck chuck"
#print("-".join(test.split()))           # prints "How-much-wood-would-a-woodchuck-chuck"



# This function converts measurement equivalents. Output is formatted 
# as, "x ounces equals y pounds", with y limited to 2 decimal places. 
# def convert_weight(ounces):

#     # Conversion formula: 1 pound = 16 ounces
#     pounds = ounces/16 
    
#     # The result is composed using the .format() method. There are two
#     # placeholders in the string: the first is for the "ounces" 
#     # variable and the second is for the "pounds" variable. The second
#     # placeholder formats the float result of the conversion 
#     # calculation to be limited to 2 decimal places.
#     result = "{} ounces equals {:.2f} pounds".format(ounces,pounds)
#     return result


# print(convert_weight(12)) # Should be: 12 ounces equals 0.75 pounds
# print(convert_weight(50.5)) # Should be: 50.5 ounces equals 3.16 pounds
# print(convert_weight(16)) # Should be: 16 ounces equals 1.00 pounds




# # This function generates a username using the first 3 letters of a
# # user’s last name plus their birth year. 
# def username(last_name, birth_year):
    
#     # The .format() method will use the first 3 letters at index 
#     # positions [0,1,2] of the "last_name" variable for the first
#     # {} placeholder. The second {} placeholder concatenates the user’s
#     #  "birth_year" to that string to form a new string username.
#     return("{}{}".format(last_name[0:3],birth_year))


# print(username("Ivanov", "1985")) 
# # Should display "Iva1985" 
# print(username("Rodríguez", "2000")) 
# # Should display "Rod2000" 
# print(username("Deng", "1991")) 
# # Should display "Den1991"
# Iva1985
# Rod2000
# Den1991

# x = ["Now", "we", "are", "cooking!"]
# len(x)
#output 4

#x = ["Now", "we", "are", "cooking!"]
#"are" in x
#output true

# x = ["Now", "we", "are", "cooking!"]
# print(x[0])
# print(x[3])
#output Now
#       Cooking

# x = ["Now", "we", "are", "cooking!"]
# x[2:]
#['are', 'cooking!']

# fruits = ["Pineapple", "Banana", "Apple", "Melon"]
# fruits.append("Kiwi")
# print(fruits)
#output ['Pineapple', 'Banana', 'Apple', 'Melon', 'Kiwi' ]

# fruits = ["Pineapple", "Banana", "Apple", "Melon"]
# fruits.insert(0, "Orange")
# print(fruits)
#['Orange', 'Pineapple', 'Banana', 'Apple', 'Melon']

# fruits = ["Pineapple", "Banana", "Apple", "Melon"]
# fruits.insert(0, "Orange")
# fruits.insert(25, "Peach")
# print(fruits)
#['Orange', 'Pineapple', 'Banana', 'Apple', 'Melon', 'Peach']

# fruits = ["Pineapple", "Banana", "Apple", "Melon"]
# fruits.insert(0, "Orange")
# fruits.insert(25, "Peach")
# fruits.remove("Melon")
# print(fruits)
#['Orange', 'Pineapple', 'Banana', 'Apple', 'Peach']


# fruits = ["Pineapple", "Banana", "Apple", "Melon"]
# fruits.insert(0, "Orange")
# fruits.insert(25, "Peach")
# fruits.remove("Melon")
# fruits.pop(3)
# fruits[2] = "Strawberry"
# print(fruits)

#['Orange', 'Pineapple', 'Strawberry', 'Peach']


# def convert_seconds(seconds):
#   hours = seconds // 3600
#   minutes = (seconds - hours * 3600) // 60
#   remaining_seconds = seconds - hours * 3600 - minutes * 60
#   return hours, minutes, remaining_seconds
# result = convert_seconds(5000)
# print(result)
#output (1, 23, 20)


# def file_size(file_info):
#     name, file_type, size = file_info
#     return "{:.2f}".format(size / 1024)

# print(file_size(('Class Assignment', 'docx', 17875))) # Should print 17.46
# print(file_size(('Notes', 'txt', 496))) # Should print 0.48
# print(file_size(('Program', 'py', 1239))) # Should print 1.21


# animals = ["Lion", "Zebra", "Dolphin", "Monkey"]
# chars = 0
# for animal in animals:
#   chars += len(animal)

# print("Total characters: {}, Average length: {}".format(chars, chars/len(animals)))

# winners = ["Ashley", "Dylan", "Reese"]
# for index, person in enumerate(winners):
#   print("{} - {}".format(index + 1, person))
#1 - Ashley
# 2 - Dylan
# 3 - Reese

# def full_emails(people):
#   result = []
#   for email, name in people:
#     result.append("{} <{}>".format(name, email))
#   return result
# print(full_emails([("alex@example.com", "Alex Diego"), ("shay@example.com", "Shay Brandt")]))

# animals = [ "lions", "zebra", "dophlin", "monkey"]
# chars = 0
# for animal in animals:
#    chars += len(animal)

# print("Total characters: {}, Average length: {}".format(chars,chars/len(animals)))
#Total characters: 23, Average length: 5.75

# names = ["anshul","ayush","sanya","ajay"]
# chars = 0
# for name in names:
#    chars = chars + len(name)

# print("Total characters:{}, Average length: {}".format(chars,chars/len(names)))
#Total characters:20, Average length: 5.0


# winners = ['John', 'Mary', 'David']

# for index, person in enumerate(winners):
#     print(f"Index: {index}, Person: {person}")

# Index: 0, Person: John
# Index: 1, Person: Mary
# Index: 2, Person: David


# By using enumerate, you can easily access both the index and the value of each item in the sequence, which can be very useful in many situations!

# multiples = []
# for x in range(1,11):
#   multiples.append(x*7)

# print(multiples)
#[7, 14, 21, 28, 35, 42, 49, 56, 63, 70]

# languages = ["Python", "Perl", "Ruby", "Go", "Java", "C"]
# lengths = [len(language) for language in languages]
# print(lengths)
#[6, 4, 4, 2, 4, 1]

# z = [x for x in range(0,101) if x % 3 == 0]
# print(z)
#[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99]

#For loop vs. list comprehension
### Simple List Comprehension
# print("List comprehension result:")

# The following list comprehension compacts several lines 
# of code into one line:
# print([x*2 for x in range(1,11)])

### Long form for loop
# print("Long form code result:")

# The list comprehension above accomplishes the same result as
# the long form version of the code shown below:
# my_list = []
# for x in range(1,11):
#     my_list.append(x*2)
# print(my_list)

# Select Run to compare the two results.



#List comprehension with conditional statement
# print("List comprehension result:")
# print([x for x in range(1,101) if x % 10 == 0])

# The list comprehension above accomplishes the same result as
# the long form version of the code:
# print("Long form code result:")
# my_list = []
# for x in range(1,101):
#     if x % 10 == 0:
#         my_list.append(x)
# print(my_list)

# Select Run to observe the two results.


# This function splits a given string into a list of elements. Then, it
# modifies each element by moving the first character to the end of the 
# element and adds a dash between the element and the moved character. 
# For example, the element "2two" will be changed to "two-2". Finally,
# the function converts the list back to a string, and returns the
# new string.



#Use map() and convert the map object to a list so we can print all the results at once.
# # A simple function to add 1 to a given number
# def add_one(number):
#     return number + 1

# # A list of numbers
# numbers = [1, 2, 3, 4, 5]

# # Use map to apply the function to each element in the list
# result = map(add_one, numbers)

# # Convert the map object to a list to print the result
# print(list(result))

# Outputs: [2, 3, 4, 5, 6]



#Use zip() to combine a list of names and ages into a list of tuples, and print all the tuples at once.
# Two lists
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

# Use zip to combine the lists
combined = zip(names, ages)

# Convert the zip object to a list to print the result
print(list(combined))

# Outputs: [('Alice', 25), ('Bob', 30), ('Charlie', 35)]




#dictionaries 

#x = {}
# type(x)
# Outputs: <class 'dict'>

# file_counts = {"jpg":10, "txt":14, "csv":2, "py":23}
# print(file_counts)
# Outputs: {'jpg': 10, 'txt': 14, 'csv': 2, 'py': 23}

# file_counts = {"jpg":10, "txt":14, "csv":2, "py":23}
# file_counts["cfg"] = 8
# print(file_counts)
# Outputs: {'jpg': 10, 'txt': 14, 'csv': 2, 'py': 23, 'cfg': 8}


# file_counts = {"jpg":10, "txt":14, "csv":2, "py":23, 'cfg':8}
# del file_counts["cfg"]
# print(file_counts)
# Outputs: {'jpg': 10, 'txt': 14, 'csv': 2, 'py': 23}


# file_counts = {"jpg":10, "txt":14, "csv":2, "py":23}
# for extension in file_counts:
#   print(extension)
#jpg
# txt
# csv
# py

# file_counts = {"jpg":10, "txt":14, "csv":2, "py":23}
# for ext, amount in file_counts.items():
#   print("There are {} files with the .{} extension".format(amount, ext))
#output There are 10 files with the .jpg extension
# There are 14 files with the .txt extension
# There are 2 files with the .csv extension
# There are 23 files with the .py extension


# file_counts = {"jpg":10, "txt":14, "csv":2, "py":23}
# file_counts.keys()
# file_counts.values()
#output dict_values([10, 14, 2, 23])

#def count_letters(text):
#   result = {}
#   for letter in text:
#     if letter not in result:
#       result[letter] = 0
#     result[letter] += 1
#   return result
# count_letters("aaaaa")
# count_letters("tenant")
# count_letters("a long string with a lot of letters")

#output-{'a': 2, ' ': 7, 'l': 3, 'o': 3, 'n': 2, 'g': 2, 's': 2, 't': 5, 'r': 2, 'i': 2, 'w': 1, 'h': 1, 'f': 1, 'e': 2}

# pet_dictionary = {"dogs": ["Yorkie", "Collie", "Bulldog"], "cats": ["Persian", "Scottish Fold", "Siberian"], "rabbits": ["Angora", "Holland Lop", "Harlequin"]}  


# print(pet_dictionary.get("dogs", 0))
# # Should print ['Yorkie', 'Collie', 'Bulldog']


# Lists only:
# are ordered sets;

# access list elements by index positions;

# require that these indices be integers;

# use square brackets [ ];

# use commas to separate each list element.



# pet_list  = ["Yorkie", "Collie", "Bulldog", "Persian", "Scottish Fold", "Siberian", "Angora", "Holland Lop", "Harlequin"]


# print(pet_list[0:3])
# Should print ['Yorkie', 'Collie', 'Bulldog']




# This function returns the total time, with minutes represented as 
# decimals (example: 1 hour 30 minutes = 1.5), for all end user time
# spent accessing a server in a given day. 


# def sum_server_use_time(Server):

#     # Initialize the variable as a float data type, which will be used
#     # to hold the sum of the total hours and minutes of server usage by
#     # end users in a day.
#     total_use_time = 0.0

#     # Iterate through the "Server" dictionary’s key and value items 
#     # using a for loop.
#     for key,value in Server.items():

#         # For each end user key, add the associated time value to the
#         # total sum of all end user use time.
#         total_use_time += Server[key]
        
#     # Round the return value and limit to 2 decimal places.
#     return round(total_use_time, 2)  

# FileServer = {"EndUser1": 2.25, "EndUser2": 4.5, "EndUser3": 1, "EndUser4": 3.75, "EndUser5": 0.6, "EndUser6": 8}

# print(sum_server_use_time(FileServer)) # Should print 20.1




# This function receives a dictionary, which contains common employee 
# last names as keys, and a list of employee first names as values. 
# The function generates a new list that contains each employees’ full
# name (First_name Last_Name). For example, the key "Garcia" with the 
# values ["Maria", "Hugo", "Lucia"] should be converted to a list 
# that contains ["Maria Garcia", "Hugo Garcia", "Lucia Garcia"].


# def list_full_names(employee_dictionary):
#     # Initialize the "full_names" variable as a list data type using
#     # empty [] square brackets.  
#     full_names = []

#     # The outer for loop iterates through each "last_name" key and 
#     # associated "first_name" values, in the "employee_dictionary" items.
#     for last_name, first_names in employee_dictionary.items():

#         # The inner for loop iterates over each "first_name" value in 
#         # the list of "first_names" for one "last_name" key at a time.
#         for first_name in first_names:

#             # Append the new "full_names" list with the "first_name" value
#             # concatenated with a space " ", and the key "last_name". 
#             full_names.append(first_name+" "+last_name)
            
#     # Return the new "full_names" list once the outer for loop has 
#     # completed all iterations. 
#     return(full_names)

# print(list_full_names({"Ali": ["Muhammad", "Amir", "Malik"], "Devi": ["Ram", "Amaira"], "Chen": ["Feng", "Li"]}))
# # Should print ['Muhammad Ali', 'Amir Ali', 'Malik Ali', 'Ram Devi', 'Amaira Devi', 'Feng Chen', 'Li Chen']




# This function receives a dictionary, which contains resource 
# categories (keys) with a list of available resources (values) for a 
# company’s IT Department. The resources belong to multiple categories.
# The function should reverse the keys and values to show which 
# categories (values) each resource (key) belongs to. 


# def invert_resource_dict(resource_dictionary):
#   # Initialize a "new_dictionary" variable as a dict data type using
#   # empty {} curly brackets. 
#     new_dictionary = {}
#     # The outer for loop iterates through each "resource_group" and 
#     # associated "resources" in the "resource_dictionary" items.
#     for resource_group, resources in resource_dictionary.items():
#         # The inner for loop iterates over each "resource" value in 
#         # the list of "resources" for one "resource_group" key at a time.
#         for resource in resources:
#             # The if-statement checks if the current "resource" value has 
#             # been appended as a key to the "new_dictionary" yet.
#             if resource in new_dictionary:
#                 # If True, then append the "resource_group" as a value to the
#                 # "resource", which is now the key.
#                 new_dictionary[resource].append(resource_group)
#             # If False (else), then add the "resource" as a new key with the 
#             # "resource_group" as a value for that key.
#             else:
#                 new_dictionary[resource] = [resource_group]
#     # Return the new dictionary once the outer for loop has completed  
#     # all iterations.
#     return(new_dictionary)


# print(invert_resource_dict({"Hard Drives": ["IDE HDDs", "SCSI HDDs"],
#         "PC Parts":  ["IDE HDDs", "SCSI HDDs", "High-end video cards", "Basic video cards"], "Video Cards": ["High-end video cards", "Basic video cards"]}))
# # Should print {'IDE HDDs': ['Hard Drives', 'PC Parts'], 'SCSI HDDs': ['Hard Drives', 'PC Parts'], 'High-end video cards': ['PC Parts', 'Video Cards'], 'Basic video cards': ['PC Parts', 'Video Cards']}



# def sales_prices(item_and_price):
#     # Initialize variables "item" and "price" as strings
#     item = ""
#     price = ""
#     # Create a variable "item_or_price" to hold the result of the split. 
#     item_or_price = item_and_price.split()

#     # For each element "x" in the split variable "item_or_price" 
#     for x in item_or_price:

#         # Check if the element is a letter
#         if x.isalpha():

#             # If true, assign the element to the "item" string variable and add a space 
#             # for any item names containing multiple words, like "Winter fleece jacket".
#             item += x + " "

#         # Else, if x is a number (if x.isalpha() is false): 
#         else:
#             # Assign the element to the "price" string variable. 
#             price = x

#     # Strip the extra space to the right of the last "item" word
#     item = item.strip()

#     # Return the item name and price formatted in a sentence 
#     return "{} are on sale for ${}".format(item,price)


# # Call to the function 
# print(sales_prices("Winter fleece jackets 49.99"))
# # Should print "Winter fleece jackets are on sale for $49.99"



# # This function accepts a string variable "data_field".  
# def count_words(data_field):

#     # Splits the string into individual words. 
#     split_data = data_field.split()
  
#     # Then returns the number of words in the string using the len()
#     # function. 
#     return len(split_data)
    
#     # Note that it is possible to combine the len() function and the 
#     # .split() method into the same line of code by inserting the 
#     # data_field.split() command into the the len() function parameters.

# # Call to the function
# print(count_words("Catalog item 3523: Organic raw pumpkin seeds in shell"))
# # Output should be 9


# # This function accepts two variables, each containing a list of years.
# # A current "recent_first" list contains [2022, 2018, 2011, 2006].
# # An older "recent_last" list contains [1989, 1992, 1997, 2001].
# # The lists need to be combined with the years in chronological order.
# def record_profit_years(recent_first, recent_last):

#     # Reverse the order of the "recent_first" list so that it is in 
#     # chronological order.
#     recent_first.reverse()

#     # Extend the "recent_last" list by appending the newly reversed 
#     # "recent_first" list.
#     recent_last.extend(recent_first)

#     # Return the "recent_last", which now contains the two lists 
#     # combined in chronological order. 
#     return recent_last

# # Assign the two lists to the two variables to be passed to the 
# # record_profit_years() function.
# recent_first = [2022, 2018, 2011, 2006]
# recent_last = [1989, 1992, 1997, 2001]
# [1989, 1992, 1997, 2001, 2006, 2011, 2018, 2022]








# def get_event_date(event):
#   return event.date

# def current_users(events):
#   events.sort(key=get_event_date)
#   machines = {}
#   for event in events:
#     if event.machine not in machines:
#       machines[event.machine] = set()
#     if event.type == "login":
#       machines[event.machine].add(event.user)
#     elif event.type == "logout":
#       machines[event.machine].remove(event.user)
#   return machines

# def generate_report(machines):
#   for machine, users in machines.items():
#     if len(users) > 0:
#       user_list = ", ".join(users)
#       print("{}: {}".format(machine, user_list))

# class Event:
#   def __init__(self, event_date, event_type, machine_name, user):
#     self.date = event_date
#     self.type = event_type
#     self.machine = machine_name
#     self.user = user

# events = [
#   Event('2020-01-21 12:45:46', 'login', 'myworkstation.local', 'jordan'),
#   Event('2020-01-22 15:53:42', 'logout', 'webserver.local', 'jordan'),
#   Event('2020-01-21 18:53:21', 'login', 'webserver.local', 'lane'),
#   Event('2020-01-22 10:25:34', 'logout', 'myworkstation.local', 'jordan'),
#   Event('2020-01-21 08:20:01', 'login', 'webserver.local', 'jordan'),
#   Event('2020-01-23 11:24:35', 'login', 'mailserver.local', 'chris'),
# ]

# users = current_users(events)


# # Call the record_profit_years() function and pass the two lists as 
# # parameters. 
# print(record_profit_years(recent_first, recent_last))
# # Should print [1989, 1992, 1997, 2001, 2006, 2011, 2018, 2022]
# print(users)

# generate_report(users)




#To print a list of all the machines that are currently in use.
# def generate_report(machines):
#   for machine, users in machines.items():
#     if len(users) > 0:
#       user_list = ", ".join(users)
#       print("{}: {}".format(machine, user_list))
