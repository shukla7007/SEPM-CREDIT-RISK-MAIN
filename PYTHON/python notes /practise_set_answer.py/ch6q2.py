marks1 = int(input("Enter the marks entered by the user:"))
marks2 = int(input("Enter the marks entered by the user:"))
marks3 = int(input("Enter the marks entered by the user:"))

total_percentage = (100*(marks1+marks2+marks3))/300

if(total_percentage>=40 and marks1>=33 and marks2>=33 and marks3>=33):
   print("Pass")

else:
   print("Fail")

if(marks1>=33):
   print("Pass in 1st subject")
else:
   print("Fail in 1st subject")
if(marks2>=33):
   print("Pass in 2nd subject")
else:
   print("Fail in 2nd subject")

if(marks3>=33):
   print("Pass in 3rd subject")
else:
   print("Fail in 3rd subject")