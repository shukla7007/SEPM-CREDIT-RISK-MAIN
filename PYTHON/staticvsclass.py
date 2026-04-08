'''we will use static method when we want to do something that should  not be unique per instance'''

class MyClass:
    # Class variable
    class_variable = "I'm a class variable"

    def __init__(self, instance_variable):
        # Instance variable
        self.instance_variable = instance_variable

    # Static method
    @staticmethod
    def static_method():
        # Static methods belong to the class, not instances
        # They can't access instance or class variables
        print("I'm a static method")
        # Trying to access instance or class variables will raise an error
        # print(self.instance_variable)  # Raises TypeError
        # print(MyClass.class_variable)  # Raises NameError

    # Class method
    @classmethod
    def class_method(cls):
        # Class methods belong to the class, not instances
        # They can access class variables, but not instance variables
        print("I'm a class method")
        print(cls.class_variable)  # Accessing class variable
        # Trying to access instance variables will raise an error
        # print(self.instance_variable)  # Raises TypeError

    # Instance method
    def instance_method(self):
        # Instance methods belong to instances, not the class
        # They can access instance and class variables
        print("I'm an instance method")
        print(self.instance_variable)  # Accessing instance variable
        print(MyClass.class_variable)  # Accessing class variable

# Creating an instance of MyClass
obj = MyClass("I'm an instance variable")

# Calling static method
MyClass.static_method()  # Output: I'm a static method
obj.static_method()  # Output: I'm a static method

# Calling class method
MyClass.class_method()  # Output: I'm a class method, I'm a class variable
obj.class_method()  # Output: I'm a class method, I'm a class variable

# Calling instance method
obj.instance_method()  # Output: I'm an instance method, I'm an instance variable, I'm a class variable
# MyClass.instance_method()  # Raises TypeError