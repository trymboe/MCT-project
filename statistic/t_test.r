E <- c(1, 1, 2, 2, 2, 2, 2, 3, 5, 6, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 1, 1, 1, 1, 2, 2, 3, 3, 3, 6, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 1, 1, 1, 2, 2, 2, 2, 2, 3, 7, 1, 2, 2, 2, 2, 3, 3, 4, 4, 6, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 1, 1, 1, 2, 2, 2, 3, 3, 3, 7, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 5, 7, 2, 2, 2, 2, 2, 3, 4, 4, 4, 6, 1, 2, 2, 2, 2, 3, 3, 4, 4, 7, 1, 2, 2, 2, 2, 2, 3, 4, 6, 7, 3, 3, 3, 4, 5, 5, 6, 6, 7, 7, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 1, 2, 2, 2, 2, 3, 3, 4, 7, 7, 1, 1, 2, 4, 4, 5, 6, 6, 7, 7, 1, 2, 2, 2, 2, 3, 4, 5, 5, 6, 1, 1, 3, 3, 3, 3, 4, 4, 4, 6, 2, 3, 3, 3, 4, 4, 6, 6, 6, 6, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 1, 1, 2, 2, 2, 2, 2, 2, 3, 7)
PR <- c(1, 2, 2, 3, 3, 3, 4, 6, 6, 7, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 1, 2, 2, 3, 4, 4, 4, 5, 5, 7, 1, 1, 1, 2, 2, 2, 2, 3, 3, 7, 1, 1, 2, 3, 3, 3, 3, 4, 5, 7, 2, 2, 2, 3, 4, 4, 5, 5, 6, 7, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 2, 2, 3, 3, 4, 4, 4, 4, 5, 6, 1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 1, 1, 1, 2, 2, 2, 2, 2, 4, 7, 1, 1, 3, 4, 4, 4, 4, 4, 5, 6, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 2, 2, 3, 4, 4, 4, 5, 5, 6, 7, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 1, 2, 2, 2, 2, 2, 3, 4, 6, 7, 1, 2, 2, 3, 4, 4, 4, 4, 6, 6, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 1, 3, 3, 3, 4, 4, 4, 4, 5, 5, 1, 3, 4, 4, 5, 5, 5, 5, 6, 7, 1, 2, 2, 2, 3, 3, 5, 6, 6, 6, 1, 2, 3, 3, 4, 4, 4, 4, 5, 7)
RP <- c(1, 1, 2, 2, 3, 3, 3, 3, 5, 6, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 1, 1, 2, 2, 2, 2, 3, 3, 3, 7, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 1, 1, 2, 2, 3, 4, 4, 4, 5, 7, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 1, 1, 1, 1, 1, 2, 3, 3, 3, 6, 1, 1, 2, 2, 2, 3, 3, 3, 3, 7, 1, 1, 3, 3, 3, 3, 3, 3, 4, 5, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 1, 3, 3, 4, 4, 4, 4, 5, 6, 7, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6, 1, 1, 2, 2, 2, 3, 3, 4, 6, 7, 1, 2, 2, 2, 2, 3, 3, 4, 6, 7, 1, 2, 2, 3, 3, 3, 3, 3, 4, 7, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 1, 1, 1, 1, 2, 2, 3, 5, 6, 6, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 1, 2, 3, 3, 3, 3, 5, 5, 6, 7, 1, 1, 1, 2, 2, 2, 3, 3, 3, 7, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 3, 5, 5, 6)

print(t.test(PR, E))
print(t.test(RP, PR))
print(t.test(E, RP))
