n = int(input("Enter the order of the matrix: "))
A = []
print("Enter the elements in the matrix:")
for i in range(n):
    row = []
    for j in range(n):
        row.append(int(input()))
    A.append(row)

m = int(input("Enter the power:")) 

result = A
 
for p in range(m-1): 
    temp = []
    for i in range(n):
        row =[]
        for j in range(n):
            s=0
            for k in range(n):
                s = s + result[i][k]*A[k][j]
            row.append(s)
        temp.append(row)
    result = temp 

print("A^", m, " = ")
for i in range(n):
    print(result[i])