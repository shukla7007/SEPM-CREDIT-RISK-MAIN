# #single inheritance 
# class Employee:#this is the base class 
#     company = "ITC"
#     def show(self):
#         print(f"name is {self.name} and the salary is {self.salary}")


# # class programmer:
# #     company = "ITC infotech"
# #     def show(self):
# #         print(f"the name is {self.name} and salary is {self.salary}")  

# #     def showLanguage(self):
# #         print(f"the name is {self.name} and he is good with {self.language} language")

# #TO MAKE CHANGES IN CLASS WE NEED TO MAKE CHANGES IN EACH CLASS ALSO AND IT CAN LEAD TO MISTAKE TO AVOID THIS MISTAKE WE USE INHERITANCE 
# #WE WILL CREATE A INHERITED CLASS 
# class Programmer(Employee):# this is the inherited class 
#     company = "ITC infotech"
#     def showLanguage(self):
#         print(f"the name is {self.name} and he is good with {self.language} language")


# a = Employee()
# b = Programmer()

# print(a.company,b.company)



#multiple inheritance 

# class Employee:#this is the base class 
#     company = "ITC"
#     name = "Default name"
#     def show(self):
#         print(f"name is {self.name} and the salary is {self.company}")

# class Coder:
#     language = "python"
#     def printLanguages(self):
#          print(f"out of all the languages here is your language: {self.language}")

# class Programmer(Employee,Coder):# this is the inherited class 
#     company = "ITC infotech"
#     def showLanguage(self):
#         print(f"the name is {self.company} and he is good with {self.language} language")


# a = Employee()
# b = Programmer()

# b.show()
# b.printLanguages()
# b.showLanguage()


#multilevel_inheritance

# class Employee:
#     a = 1
# class Programmer(Employee):
#     b = 2

# class Manager(Programmer):
#     c = 3

# o = Employee()
# print(o.a)
# # print(o.b)##it will show error AttributeError: 'Employee' object has no attribute 'b' 

# o = Programmer()
# print(o.a,o.b)

# o = Manager()
# print(o.a,o.b,o.c)

#SUPER.PY

# class Employee:
#     def __init__(self) -> None:
#         print("Constructor of Employee")
#     a = 1
# class Programmer(Employee):
#     def __init__(self) -> None:
#         print("Constructor of Programmer")
#     b = 2

# class Manager(Programmer):
#     def __init__(self) -> None:
#         super().__init__()#when we to call constructor of programmer also with the manager
#         print("Constructor of Manager")
#     c = 3

# o = Employee()
# print(o.a)
# # print(o.b)##it will show error AttributeError: 'Employee' object has no attribute 'b' 

# o = Programmer()
# print(o.a,o.b)

# o = Manager()
# print(o.a,o.b,o.c)


#class attribute 
# class Employee:
#     a = 1
#     @classmethod
#     def show(cls):
#         print(f"the class attribute of a is {cls.a} ")
# e = Employee()
# e.a = 45

# e.show()#it showing the class attribute of a is 45 but when we use @classmethod and replace self with cls then the attribute of a is 1


#property decorators 
# class Employee:
#     a = 1  # Class attribute
    
#     @classmethod
#     def show(cls):
#         print(f"The class attribute 'a' is {cls.a}")

#     def __init__(self):
#         self._name = None  # Instance attribute for name
    
#     # Getter method using @property
#     @property
#     def name(self):
#         return self._name
    
#     # Setter method using @property
#     @name.setter
#     def name(self, value):
#         self.fname = value.split(" ")[0]
#         self.lname = value.split(" ")[1]
#         self._name = value


# e = Employee()

# # Setting the instance attribute 'a' for the object e
# e.a = 45  # This creates an instance attribute 'a' for e, separate from the class attribute 'a'

# # Using the property setter for 'name'
# e.name = "Anshul Shukla"

# # Accessing the name using the property getter
# print(e.name)  # Output: Anshul Shukla

# # Showing the class attribute 'a' using the class method
# e.show()  # Output: The class attribute 'a' is 1


#n Python, the @property decorator is used to create getters and setters in an elegant and Pythonic way. It allows you to define methods in a class that can be accessed like attributes, offering encapsulation and control over the access and modification of data attributes.
#The @property decorator is used to create a getter method, and the @x.setter decorator is



#operator overloading 

class Number:
    def __init_subclass__(self,n):
        self.n = n
    def __add__(self,num):#as wee need to specify what plus does 
        return self.n+num.n

n = Number(1)
m = Number(2) 

print(n+m)#it will show error as everything is class in python we need to define what plus do


