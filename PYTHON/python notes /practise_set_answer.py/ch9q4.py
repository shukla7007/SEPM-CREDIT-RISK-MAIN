word = "Donkey"

with open("file.txt" "r") as f:
    content = f.read()

contentNew = content.replace("Donkey", "######")

