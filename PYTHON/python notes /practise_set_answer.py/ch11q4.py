class Complex:

    def __init__(self, r, i):
        self.r = r  # real part
        self.i = i  # imaginary part

    # Addition of two complex numbers
    def __add__(self, c2):
        return Complex(self.r + c2.r, self.i + c2.i)

    # Multiplication of two complex numbers
    def __mul__(self, c2):
        real_part = self.r * c2.r - self.i * c2.i  # ac - bd
        imaginary_part = self.r * c2.i + self.i * c2.r  # ad + bc
        return Complex(real_part, imaginary_part)
   
    # String representation of the complex number
    def __str__(self):
        return f"{self.r} + {self.i}i"
    

# Example usage
c1 = Complex(1, 2)  # 1 + 2i
c2 = Complex(3, 4)  # 3 + 4i

# Testing addition
print("Addition:", c1 + c2)  # Should print: 4 + 6i

# Testing multiplication
print("Multiplication:", c1 * c2)  # Should print: -5 + 10i
