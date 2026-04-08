f = open("ch9q1poems.txt")
content = f.read()
if("twinkle" in content):
    print("twinkle is present")
else:
    print("the word twinkle is not present in the content")

f.close()