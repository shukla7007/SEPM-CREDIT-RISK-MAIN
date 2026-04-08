n = int(input("Enter the number:"))
i=1
for i in range (1,n+1):
  print(' '*(n-i),end='')#otherwise it will give new line you don't want to new line to form pattern 
  print('*' * (2*i-1),end='')
  print('')
