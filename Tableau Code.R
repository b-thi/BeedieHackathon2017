# Business Hackathon- Brad Code
# Tableau Code

install.packages("tidyverse")
install.packages("Rserve")
install.packages("stringi")
install.packages("xgboost")
install.packages("caret")


library(Rserve)
library(tidyverse)
library(stringi)
library(xgboost)
library(caret)


# Data
data = read.csv("Retention2017.csv", as.is = TRUE, strip.white = TRUE, header = TRUE)
View(data)
dim(data)

data %>% 
  filter(Sample == "Estimation") -> train


# Set up R Server
Rserve()



# Testing a basic model
names(train)

as.factor(train$lost)

fit1 <- glm(as.factor(lost) ~ eopenrate + eclickrate + avgorder + ordfreq, data = train, family = binomial)

fit1


