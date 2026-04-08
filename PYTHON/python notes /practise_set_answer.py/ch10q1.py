class Programmer:
    company = "Microsoft"
    def __init__(self, name, salary, pin):
         self.name = name 
         self.salary = salary
         self.pin = pin
    

p = Programmer("Anshul Shukla",3600000,121002)
print(p.name,p.salary,p.pin,p.company)

r = Programmer("Ayush Shukla",2500000,200302)
print(r.name,r.salary,r.pin,r.company)