word = "hippopotamus"

max_count = 0
max_char = ""

for i in range(len(word)):
    count = 0
    for j in range(len(word)):
        if word[i] == word[j]:
            count += 1

    if count > max_count:
        max_count = count
        max_char = word[i]

print("Highest occurring character:", max_char)
print("Occurrence count:", max_count)
