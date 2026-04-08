class Demo:
    a = 4

o = Demo()
print(o.a)#present the class attribute because instance attribute is not present 
o.a = 0 #instance attribute is set
print(o.a)#prints the instance attribute beacuse instance attribute is present  
print(Demo.a)

#it does not change class attribute 

