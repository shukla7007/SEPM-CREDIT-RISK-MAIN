class twoDV:
    def __init__(self,i,j):
        self.i = i
        self.j = j


    def show(self):
      print(f"the vecotr is {self.i}.i + {self.j}.j ")



class threeDV(twoDV):
    def __init__(self,i,j,k):
        super().__init__(i,j)#so that it will get set when above class runs 
        self.k = k
    def show(self):
      print(f"the vecotr is {self.i}.i + {self.j}.j + {self.k}.k")

a = twoDV(1,2)
a.show()
b =  threeDV(1,2,3)
b.show()
