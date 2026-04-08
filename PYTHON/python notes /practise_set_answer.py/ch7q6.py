n = int(input("Enter the number:"))

i=1
product = 1
for i in range(1,n+1):
    product = product * i
    i = i+1

print(f"the factorial of {n} is {product}")