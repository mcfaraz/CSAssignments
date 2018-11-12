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
quadWeight = lm(formula = abaloneData$WholeWeight ~ abaloneData$Diameter + I(abaloneData$Diameter^2) + I(abaloneData$Diameter^3))
cubeWeight = lm(formula = abaloneData$WholeWeight ~ I(abaloneData$Diameter^3) + 0)
expWeight = lm(formula = log(abaloneData$WholeWeight) ~ abaloneData$Diameter)

pred_lm = predict(lmWeight)
pred_quad = predict(quadWeight)
pred_cube = predict(cubeWeight)
pred_exp = predict(expWeight)

plot(abaloneData$WholeWeight ~ abaloneData$Diameter)
lines(abaloneData$Diameter, y=pred_lm, col = "green")
lines(abaloneData$Diameter, y=pred_quad, col = "blue")
lines(abaloneData$Diameter, y=pred_cube, col = "red")
lines(abaloneData$Diameter, y=exp(pred_exp), col = "yellow")
