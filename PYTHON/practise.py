# def sieve(n):
#     if n <= 1:
#         return []
    
#     isPrime = [True] * (n + 1)
#     isPrime[0] = isPrime[1] = False  # 0 and 1 are not prime numbers

#     i = 2
#     while i * i <= n:
#         if isPrime[i]:
#             for j in range(i * i, n + 1, i):  # Start from i^2
#                 isPrime[j] = False
#         i += 1

#     primes = [i for i in range(2, n + 1) if isPrime[i]]
#     return primes  # Return the list instead of printing

# if __name__ == '__main__':
#     n = int(input("Enter the number: "))
#     print(sieve(n))  # Print the result outside the function

# def remdup(arr,n):
#     temp = [0] * n
#     temp[0] = arr[0]
#     res = 1
#     for i in range(1,n):
#         if temp[res-1]! = arr[i]:
#             temp[res] = arr[i]
#             res += 1
#     for i in range(0,res):
#         arr[i] = temp[i]
#         return res

x = int(input("Enter the number :"))

if x == 0:
    res = 1
else:
     x = abs(x)
     res = 0

while x > 0:
      x = x // 10
      res += 1

print(res)  # Print the result outside the function
