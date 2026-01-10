import random

number = []

for i in range(25):
    number.append(random.randint(1,10))

print("Numbers:",number)

total = 0
for i in number:
    total +=i
    mean = total/25

number.sort()
median = number[12]

max= 0 
mode = number[0]

for i in number:
    count = 0
    for j in number:
        if i == j:
            count += 1
    if count > max:
        max = count
        mode = i    

print("Mean is:", mean)
print("Median is:", median)
print("Mode is:", mode)


