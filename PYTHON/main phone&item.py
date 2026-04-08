'''Instead of creating a child class and parent class, you can directly import the Item and Phone classes from their respective modules and use them to instantiate objects.'''


from item import Item
from phone import Phone


Item.instantiate_from_csv()

print(Item.all)



'''a Person class with a name attribute. We use the @property decorator to define a getter method for the name attribute, and the @name.setter decorator to define a setter method for the name attribute. The setter method checks if the length of the value parameter is greater than 10, and raises an exception if it is.'''

''' Getter and Setter Methods
Getter Method
Purpose: A getter method is used to access (or "get") the value of a private attribute from a class.
Usage: It allows controlled access to the attribute, ensuring that the attribute's value is not directly exposed or modified outside the class.
Example: In the context of the Person class, the @property decorator is used to define a getter method for the name attribute. This allows you to retrieve the value of name without directly accessing the private attribute.
Setter Method
Purpose: A setter method is used to modify (or "set") the value of a private attribute in a class.
Usage: It provides a way to enforce rules or constraints when setting the value of an attribute. For example, you can validate the input before assigning it to the attribute.
Example: In the Person class, the @name.setter decorator is used to define a setter method for the name attribute. This method checks if the length of the value parameter is greater than 10 and raises an exception if it is, ensuring that the name attribute adheres to a specific rule.
Why Use Getter and Setter to Protect Read-Only Attributes?
Encapsulation
Protection: By using getter and setter methods, you can encapsulate the internal state of an object, protecting it from unauthorized access or modification.
Control: You can control how the attribute is accessed or modified, ensuring that any changes to the attribute follow specific rules or constraints.
Read-Only Attributes
Immutable Access: If you want an attribute to be read-only (i.e., it can be accessed but not modified), you can define a getter method without a corresponding setter method. This ensures that the attribute's value cannot be changed after it is set.
Example: If the name attribute in the Person class were intended to be read-only, you would only define the getter method using @property and omit the setter method. This would prevent any external code from modifying the name attribute.
Validation and Constraints
Data Integrity: Setter methods allow you to enforce validation rules or constraints when setting the value of an attribute. This ensures that the attribute always holds valid data.
Example: In the Person class, the setter method for the name attribute ensures that the name does not exceed 10 characters, maintaining data integrity.
Summary
Getter: Provides controlled access to an attribute.
Setter: Allows controlled modification of an attribute with validation.
Read-Only: By omitting the setter, you can make an attribute read-only, protecting it from unintended modifications
'''