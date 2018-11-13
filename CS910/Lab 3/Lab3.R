library('farff')
abaloneData <- readARFF('abalone.arff')

#Q1
lm_diamter_length = lm(abaloneData$Diameter ~ abaloneData$Length)
summary(lm_diamter_length)

#Q2
colnames(abaloneData)[5] <- 'WholeWeight'
colnames(abaloneData)[6] <- 'ShuckedWeight'
colnames(abaloneData)[7] <- 'VisceraWeight'
colnames(abaloneData)[8] <- 'ShellWeight'

lm_whole_weight = lm(abaloneData$WholeWeight ~ abaloneData$ShuckedWeight + abaloneData$VisceraWeight + abaloneData$ShellWeight)
summary(lm_whole_weight)

#Q3
lmWeight = lm(formula = abaloneData$WholeWeight ~ abaloneData$Diameter)
quadWeight = lm(formula = abaloneData$WholeWeight ~ abaloneData$Diameter + I(abaloneData$Diameter^2))
cubeWeight = lm(formula = abaloneData$WholeWeight ~ I(abaloneData$Diameter^3) + 0)
expWeight = lm(formula = log(abaloneData$WholeWeight) ~ abaloneData$Diameter)

pred_lm = predict(lmWeight)
pred_quad = predict(quadWeight)
pred_cube = predict(cubeWeight)
pred_exp = predict(expWeight)

plot(abaloneData$WholeWeight ~ abaloneData$Diameter, xlab='Diameter', ylab = 'Whole Weight')
lines(abaloneData$Diameter, y=pred_lm, col = "green")
lines(abaloneData$Diameter, y=pred_quad, col = "blue")
lines(abaloneData$Diameter, y=pred_cube, col = "red")
lines(abaloneData$Diameter, y=exp(pred_exp), col = "yellow")
legend('topleft', legend=c("Original weight", "linear model", 'quadratic model', 'cubic model', 'exponential model'), col=c('black', 'green', "blue", "red", 'yellow'), lty = 1, cex=0.8)

#Q4
abaloneData <- transform(abaloneData, Age=ifelse(Sex=="I","I","A"))

#Length Only
lfit_length <- glm(abaloneData$Age ~ abaloneData$Length, family="binomial")
pairs_length <- (paste(round(predict(lfit_length, type="response")), abaloneData$Age))
a = table(pairs_length)
cat("Accuracy Length: ",(a[1]+a[4])/nrow(abaloneData))

#Whole Weight Only
lfit_whole_weight <- glm(abaloneData$Age ~ abaloneData$WholeWeight, family="binomial")
pairs_whole_weight <-(paste(round(predict(lfit_whole_weight, type="response")), abaloneData$Age))
a = table(pairs_whole_weight)
cat("Accuracy Whole Weight: ",(a[1]+a[4])/nrow(abaloneData))

#Class Rings Only
lfit_rings <- glm(abaloneData$Age ~ abaloneData$Class_Rings, family="binomial")
pairs_rings <-(paste(round(predict(lfit_rings, type="response")), abaloneData$Age))
a = table(pairs_rings)
cat("Accuracy Class Rings: ",(a[1]+a[4])/nrow(abaloneData))

#Length, whole weight, and class rings
lfit_LWC <- glm(abaloneData$Age ~ abaloneData$Length + abaloneData$WholeWeight + abaloneData$Class_Rings, family="binomial")
pairs_LWC <-(paste(round(predict(lfit_LWC, type="response")), abaloneData$Age))
a = table(pairs_LWC)
cat("Accuracy LWC: ",(a[1]+a[4])/nrow(abaloneData))

