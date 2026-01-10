my_list=[2,7,4,1,3,6]
sum=0
for i in my_list:
    for j in my_list:
        if (i+j) == 10:
            print(i,j)
            sum +=1
print(sum)
