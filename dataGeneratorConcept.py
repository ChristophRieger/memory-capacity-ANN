from itertools import product

# Generating all permutations with repetition for a sequence of length 4
possible_values = [0, 1]
length = 6
sparsity = 0.5
perm = list(product(possible_values, repeat=length))

# now remove all rows that do not sum up to 2!!!
filteredPerms = [row for row in perm if sum(row) == length * sparsity]
print("All permutations with repetition for a sequence of set length and sparsity")
for p in filteredPerms:
    print(p)