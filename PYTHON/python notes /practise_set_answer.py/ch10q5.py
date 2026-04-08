from random import randint

class Train:

    def __init__(self, trainNo):
        self.trainNo = trainNo

    def book(self, fro, to):
        print(f"Ticket is booked in train no: {self.trainNo} from {fro} to {to}")

    def getStatus(self):
        print(f"Train no: {self.trainNo} is running on time")

    def getFare(self, fro, to):
        print(f"Ticket fare in train no: {self.trainNo} from {fro} to {to} is {randint(222, 5555)}")

# Create an instance of Train
t = Train(12399)
t.book("Rampur", "Delhi")
t.getStatus()
t.getFare("Rampur", "Delhi")


#for question number 6 it does not effect the output the output still remains the  same 