args <- commandArgs(trailingOnly = TRUE)

expected = read.csv(args[1])
obtained = read.csv(args[2])

# gets the number of correct classifications
freqs = table(expected$Prediction == obtained)

# correct / total
score = (freqs["TRUE"] / sum(freqs))

sprintf("%.3f", score)
print(table(obtained))
