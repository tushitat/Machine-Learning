my_list = [5,3,8,1,0,4]

if len(my_list) < 3:
    print("Range determination not possible")

else:
    max = max(my_list)
    min = min(my_list)

    print("Range is:", max - min)
    