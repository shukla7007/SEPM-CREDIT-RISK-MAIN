n = int(input("Enter the given number:"))

for i in range(2,n):#so it is checking from 2 to n-1 as in prime 1 and itself will give remainder 0
    if n%i==0:
        print(n,"is not a prime number")
        break
else:
    print(n,"is a prime number")

