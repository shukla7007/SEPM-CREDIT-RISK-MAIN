class Employee:
    language = "Python"  # This is a class attribute
    salary = 1200000

    def __init__(self, name, salary, language):  # Dunder method which is automatically called
        self.name = name
        self.salary = salary
        self.language = language
        print("I am creating an object")

    def getInfo(self):
        print(f"The language is {self.language}. The salary is {self.salary}")

    @staticmethod
    def greet():
        print("Good morning")

# Create instances of Employee
harry = Employee("Harry", 1300000, "JavaScript")
print(harry.name, harry.salary)

rohan = Employee("Rohan", 1500000, "Python")
print(rohan.name, rohan.language)

# Call methods
harry.getInfo()
rohan.getInfo()

# Call static method
Employee.greet()
 

 #These attributes (self.name, self.salary, and self.language) are unique to each object (instance) of the class. So, harry and rohan will have different values for these attributes, even though they both belong to the Employee class.
# This ensures that each Employee object can store its own name, salary, and language independent of other Employee objects.
