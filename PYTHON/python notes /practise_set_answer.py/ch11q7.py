class Vector:
    def __init__(self, l):
        # Assign components from the list `l`
        self.x, self.y, self.z = l

    # Vector addition
    def __add__(self, other):
        result = Vector([self.x + other.x, self.y + other.y, self.z + other.z])
        return result

    # Vector multiplication (dot product)
    def __mul__(self, other):
        result = self.x * other.x + self.y * other.y + self.z * other.z
        return result

    # String representation of the vector
    def __str__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"

    # Override the __len__ method to return the dimension of the vector
    def __len__(self):
        # Since the vector always has 3 components, return 3
        return 3

# Test the implementation
v1 = Vector([1, 2, 3])

# Checking the dimension of the vector
print(len(v1))  # Output: 3
