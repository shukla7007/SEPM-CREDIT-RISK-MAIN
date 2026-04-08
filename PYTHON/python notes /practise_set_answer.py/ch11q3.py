class Employee:
    salary = 234
    _increment = 20  # Use a protected variable to manage increment

    # Property to calculate salary after increment
    @property
    def salaryAfterIncrement(self):
        return self.salary + self.salary * (self._increment / 100)

    # Getter for 'increment'
    @property
    def increment(self):
        return self._increment

    # Setter for 'increment'
    @increment.setter
    def increment(self, new_salary):
        self._increment = ((new_salary / self.salary) - 1) * 100  # Adjust increment based on new salary

#new salary = old salary (1+increment/100)
#(new salary/old salary -1)*100 = increment 


e = Employee()

# Get the salary after increment (initial increment is 20%)
print(e.salaryAfterIncrement)  # Output: 280.8 (234 + 20% of 234)

# Set a new salary and calculate the corresponding increment
e.increment = 280.8

# Get the new increment after setting the new salary
print(e.increment)  # Output: 20.0
