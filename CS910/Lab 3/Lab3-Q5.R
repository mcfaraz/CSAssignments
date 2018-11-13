library('farff')
adultsData <- readARFF('adult.arff')

for (i in names(adultsData))
{
  if (i == 'sex' | i == 'relationship' | i == 'hoursPerWeek' | i == 'maritalStatus' | i == 'education' | i == 'class' | i == 'fnlwgt' | i == 'age')
  {
    next()
  }
  fmla <- as.formula(paste0('sex ~ relationship + hoursPerWeek + maritalStatus + education + class + fnlwgt + age + ', i))
  lfit_all <- glm(fmla, data=adultsData, family='binomial')
  pairs_all <-(paste(round(predict(lfit_all, type='response')), adultsData$sex))
  table(pairs_all)
  a <- table(pairs_all)
  cat('sex ~ relationship + hoursPerWeek + maritalStatus + education + class + fnlwgt + age + ',i,' : ',100*(a[1]+a[4])/nrow(adultsData),'%\n')
}

