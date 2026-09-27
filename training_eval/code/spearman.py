from scipy.stats import spearmanr

# beir = [1, 2, 8, 3, 4, 5, 6, 7, 9]
# nanobeir = [2, 3, 7, 1, 4, 5, 6, 8, 9]
# beir = [1, 8, 6, 2, 3, 9, 5, 7, 4]
# nanobeir = [2, 7, 3, 1, 5, 8, 4, 6, 9]

# beir = [8, 10, 15, 11, 12, 13, 14, 16, 1, 7, 5, 2, 3, 9, 4, 6]

# nanobeir = [9, 10, 14, 11, 12, 13, 15, 16, 2, 7, 3, 1, 5, 8, 4, 6]

beir = [9, 11, 16, 12, 13, 14, 15, 17, 18, 1, 8, 5, 2, 3, 10, 4, 7, 6]
nanobeir = [10, 12, 15, 8, 13, 14, 16, 17, 18, 2, 7, 3, 1, 5, 9, 4, 6, 11]

rho, p_value = spearmanr(beir, nanobeir)

print("Spearman rho:", rho)
print("p-value:", p_value)
