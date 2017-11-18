# Matt's stuff for Beedie Hackathon

data = read.csv("Retention2017.csv", as.is = TRUE, strip.white = TRUE, header = TRUE)
dim(data)
head(data)

# Impute the missing data

# eopenrate, refill, doorstep have missing values

# Remove the few missing Y/N rows
data = data[!is.na(data$doorstep) & !is.na(data$refill),]
# Impute with exponential approximation for the eclickrate
rate = 1/ mean(data$eclickrate, na.rm = TRUE)
data$eclickrate[is.na(data$eclickrate)] = rexp(1, rate = rate )


Holdout = data[data$Sample == "Holdout",]
Train = data[data$Sample == "Estimation",]
Test = data[data$Sample == "Validation",]

View(head(Train))


