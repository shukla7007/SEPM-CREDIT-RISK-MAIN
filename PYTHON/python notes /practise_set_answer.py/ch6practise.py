age = int(input("Enter your age: "))

if(age%2==0):#both if statement will run idependently of each other
    print("Your age is even")

if(age>=18):
    print("You are an adult")

elif(age==0):
    print("You are a new born baby")
else:
    print("You are a minor")


print("End of program")
