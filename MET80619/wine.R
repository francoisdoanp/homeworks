####################################
#       DEVOIR - WINE DATASET      #
#  FRANCOIS DOAN-POPE - 11178569   #
####################################

# Import librairies
library(ggplot2)
library(GGally)
library(nnet)
library(MASS)
library(rpart)
library(rpartScore)
library(glmnet)
library(caret)
library(ordinalNet)
library(randomForest)

# Reproductibilit?
set.seed(1234)

# Import des fichiers

datrain <- read.table('https://raw.githubusercontent.com/francoisdoanp/homeworks/master/datrain.txt', header=TRUE, sep=' ')
dateststudent <- read.table('https://raw.githubusercontent.com/francoisdoanp/homeworks/master/dateststudent.txt', header=TRUE, sep=' ')

#Analyse exploratoire

summary(datrain)
ggplot(datrain, aes(x=residualsugar)) + geom_histogram(binwidth=1, color="blue",fill="grey")
ggplot(datrain, aes(x=y)) + geom_histogram(binwidth=1, color="blue",fill="grey") #2 est la classe la plus probable, ce sera notre baseline
colSums(sapply(datrain, is.na)) # valeurs manquantes?
corr = cor(datrain)
ggcorr(datrain,palette = "RdBu", label=TRUE)
ggpairs(datrain, columns=1:ncol(datrain), title="",
        axisLabels="show", columnLabels = colnames(datrain))

# Division de l'echantillon - 70% train, 30% test

datrain$y = as.factor(datrain$y)
sample_size <- floor(0.70 * nrow(datrain))
train_index <- sample(seq_len(nrow(datrain)), size=sample_size)
train <- datrain[train_index,]
test <- datrain[-train_index,]

# Transformation en num?rique pour certain mod?les
train_int <- train
test_int <- test
train_int$y <- as.numeric(train_int$y)
test_int$y <- as.numeric(test_int$y)



################################################
# Regression multinomiale vs regression ordinale

multiModel = multinom(y ~., data=train)
summary(multiModel)
pred_multi = predict(multiModel, test)
confusionMatrix(pred_multi, test$y)

ord_model = polr(y ~., data=train)
pred_ord = predict(ord_model, test)
confusionMatrix(pred_ord, test$y)

################################################
# Multinomial Elastic Net Vs Ordinal Net


# Transformation en matrices et vecteurs
trainy_vec = train_int$y
testy_vec  = test_int$y
trainx_mat = data.matrix(train_int[1:11])
testx_max  = data.matrix(test_int[1:11])


Melastic = cv.glmnet(trainx_mat, trainy_vec, family="multinomial", alpha=0.5)
pred_Melastic = predict(Melastic, testx_max, s= Melastic$lambda.min, type='class')
table(pred_Melastic, testy_vec)
mean((pred_Melastic) == (testy_vec))

trainy_fact = train$y
testy_fact  = test$y

tunefit = ordinalNetTune(trainx_mat, trainy_fact, alpha=0.5)
bestLambdaIndex <- which.min(rowMeans(tunefit$misclass))
coef(tunefit$fit, whichLambda = bestLambdaIndex, matrix=TRUE)
pred_Oelastic = predict(tunefit$fit, testx_max, whichLambda = bestLambdaIndex, type='class')
table(pred_Oelastic, testy_vec)
mean((pred_Oelastic) == (testy_vec))


################################################
# Arbres de classification vs Arbre de classification ordinal

# Arbre classification
start.time <- Sys.time()
ctree = rpart(y ~., data=train, method='class')
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

predCTree = predict(ctree, newdata=test, type='class')
confusionMatrix(predCTree, test$y)

# Arbre classification ordinal
start.time <- Sys.time()
otree = rpartScore(y ~., data=train_int)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

predOTree = predict(otree, newdata=test_int)
table(predOTree, test_int$y)
mean(as.character(predOTree) == as.character(test_int$y))


################################################
# Random forest vs ordinal random forest

start.time <- Sys.time()
rf <- train(y~., data=train, method="rf", metric="Accuracy")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
pred_rf = predict(rf, newdata=test)
confusionMatrix(pred_rf, test$y)

start.time <- Sys.time()
rf_or <- train(y~., data=train, method="ordinalRF", metric="Accuracy")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

pred_rfor = predict(rf_or, newdata=test)
confusionMatrix(pred_rfor, test$y)

################################################
# Tuning RF

control_rf = trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(1:11))
rf_gridsearch <- train(y~., data=datrain, method="rf", tuneGrid=tunegrid, trControl=control_rf, metric="Accuracy")

pred_final <- predict(rf_gridsearch, newdata=dateststudent)
summary(pred_final)


write.table(pred_final, file="predictions.txt", col.names = FALSE, quote=FALSE, row.names=FALSE)

write.table(pred_final, file="predictionst.txt")
