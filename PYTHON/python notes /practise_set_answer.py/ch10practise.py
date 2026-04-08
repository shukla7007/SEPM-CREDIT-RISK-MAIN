# class Employee:
#     name = "Harry"
#     language = "Py"
#     salary = 1200000
# #to create a object 
# harry = Employee()
# harry.name = "Harry"
# print(harry.name,harry.language,harry.salary)#salary and language are class attributes as it belong directly to the class

# rohan = Employee()
# rohan.name = "Rohan"
# rohan.language = "javascript"
# print(rohan.name,rohan.language,rohan.salary)#salary and language are class attributes as it belong directly to the clarohan


#here name is object/instance attribute and salary and language are class atttribute

#it will javascript now instead of py for rohan as we have made an object attribute so it will take prefrence compare to  class attribute 


class Employee:
    language = "python"
    salary = "12000"

    def __init__(self):#dunder method which is automatically called when object is created
        print("I am creating an object") 
    
    def getInfo(self):#in this function we need to write self for attribute 
        print(f"the language is {self.language}.the slary is {self.salary} ")
    
    @staticmethod#as we do not need no attribute in this to avoid using  self function
    def greet():
        print("Good morning")
harry = Employee()
harry.name= "Employee"
harry.greet()
harry.getInfo()#here we are calling the class method using the object of the class
print(harry.name,harry.salary)

rohan = Employee()

# Why is self used?
# Refers to the instance: self allows you to refer to the instance's attributes (like self.language and self.salary) and methods within the class.
# Distinguishes instance and class variables: You can access and modify instance-specific attributes using self. Without self, Python wouldn’t know whether you’re referring to a class attribute, a global variable, or an instance attribute

#when ever we create an object dunder method will be automatically called 

