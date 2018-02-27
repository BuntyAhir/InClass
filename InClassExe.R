library(tidyverse)
library(caret)
library(mlbench)

data("Glass")
Glass <- as.tibble(Glass)
head(Glass)

ggplot(data = Glass, mapping = aes(x = Type, y = RI)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = Na)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = Mg)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = Al)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = Si)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = K)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = Ca)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = Ba)) + geom_boxplot()
ggplot(data = Glass, mapping = aes(x = Type, y = Fe)) + geom_boxplot()

set.seed(1)
trainIndex <- createDataPartition(Glass$Type, p = 0.8, list = FALSE, times = 1)

Glass[trainIndex, ]
GlassTrain <- Glass[trainIndex, ]

Glass[-trainIndex, ]
GlassTest <- Glass[-trainIndex, ]

preProcess(GlassTrain, method = c("center", "scale"))
scaler <- preProcess(GlassTrain, method = c("center", "scale"))

predict(scaler, GlassTrain)
GlassTrain <- predict(scaler, GlassTrain)
predict(scaler, GlassTest)
GlassTest <- predict(scaler, GlassTest)
head(GlassTrain)

train(Type ~ RI, data = GlassTrain, method = "knn")
knn_model_RI <- train(Type ~ RI, data = GlassTrain, method = "knn")
predict(knn_model_RI, GlassTest)
GlassTestPredictions_RI <- predict(knn_model_RI, GlassTest)
confusionMatrix(GlassTestPredictions_RI, GlassTest$Type)


train(Type ~ Na, data = GlassTrain, method = "knn")
knn_model_Na <- train(Type ~ Na, data = GlassTrain, method = "knn")
predict(knn_model_Na, GlassTest)
GlassTestPredictions_Na <- predict(knn_model_Na, GlassTest)
confusionMatrix(GlassTestPredictions_Na, GlassTest$Type)

train(Type ~ Mg, data = GlassTrain, method = "knn")
knn_model_Mg <- train(Type ~ Mg, data = GlassTrain, method = "knn")
predict(knn_model_Mg, GlassTest)
GlassTestPredictions_Mg <- predict(knn_model_Mg, GlassTest)
confusionMatrix(GlassTestPredictions_Mg, GlassTest$Type)

train(Type ~ Al, data = GlassTrain, method = "knn")
knn_model_Al <- train(Type ~ Al, data = GlassTrain, method = "knn")
predict(knn_model_Al, GlassTest)
GlassTestPredictions_Al <- predict(knn_model_Al, GlassTest)
confusionMatrix(GlassTestPredictions_Al, GlassTest$Type)


train(Type ~ Si, data = GlassTrain, method = "knn")
knn_model_Si <- train(Type ~ Si, data = GlassTrain, method = "knn")
predict(knn_model_Si, GlassTest)
GlassTestPredictions_Si <- predict(knn_model_Si, GlassTest)
confusionMatrix(GlassTestPredictions_Si, GlassTest$Type)

train(Type ~ K, data = GlassTrain, method = "knn")
knn_model_K <- train(Type ~ K, data = GlassTrain, method = "knn")
predict(knn_model_K, GlassTest)
GlassTestPredictions_K <- predict(knn_model_K, GlassTest)
confusionMatrix(GlassTestPredictions_K, GlassTest$Type)


train(Type ~ Ba, data = GlassTrain, method = "knn")
knn_model_Ba <- train(Type ~ Ba, data = GlassTrain, method = "knn")
predict(knn_model_Ba, GlassTest)
GlassTestPredictions_Ba <- predict(knn_model_Ba, GlassTest)
confusionMatrix(GlassTestPredictions_Ba, GlassTest$Type)


train(Type ~ Fe, data = GlassTrain, method = "knn")
knn_model_Fe <- train(Type ~ Fe, data = GlassTrain, method = "knn")
predict(knn_model_Fe, GlassTest)
GlassTestPredictions_Fe <- predict(knn_model_Fe, GlassTest)
confusionMatrix(GlassTestPredictions_Fe, GlassTest$Type)

train(Type ~ RI + Na, data = GlassTrain, method = "knn")
knn_model_RI_Na <- train(Type ~ RI + Na, data = GlassTrain, method = "knn")
predict(knn_model_RI_Na, GlassTest)
GlassTestPredictions_RI_Na <- predict(knn_model_RI_Na, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Na, GlassTest$Type)

train(Type ~ RI + Mg, data = GlassTrain, method = "knn")
knn_model_RI_Mg <- train(Type ~ RI + Mg, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg, GlassTest)
GlassTestPredictions_RI_Mg <- predict(knn_model_RI_Mg, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg, GlassTest$Type)

train(Type ~ RI + Al, data = GlassTrain, method = "knn")
knn_model_RI_Al <- train(Type ~ RI + Al, data = GlassTrain, method = "knn")
predict(knn_model_RI_Al, GlassTest)
GlassTestPredictions_RI_Al <- predict(knn_model_RI_Al, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Al, GlassTest$Type)


train(Type ~ RI + Si, data = GlassTrain, method = "knn")
knn_model_RI_Si <- train(Type ~ RI + Si, data = GlassTrain, method = "knn")
predict(knn_model_RI_Si, GlassTest)
GlassTestPredictions_RI_Si <- predict(knn_model_RI_Si, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Si, GlassTest$Type)


train(Type ~ RI + K, data = GlassTrain, method = "knn")
knn_model_RI_K <- train(Type ~ RI + K, data = GlassTrain, method = "knn")
predict(knn_model_RI_K, GlassTest)
GlassTestPredictions_RI_K <- predict(knn_model_RI_K, GlassTest)
confusionMatrix(GlassTestPredictions_RI_K, GlassTest$Type)


train(Type ~ RI + Ca, data = GlassTrain, method = "knn")
knn_model_RI_Ca <- train(Type ~ RI + Ca, data = GlassTrain, method = "knn")
predict(knn_model_RI_Ca, GlassTest)
GlassTestPredictions_RI_Ca <- predict(knn_model_RI_Ca, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Ca, GlassTest$Type)

train(Type ~ RI + Ba, data = GlassTrain, method = "knn")
knn_model_RI_Ba <- train(Type ~ RI + Ba, data = GlassTrain, method = "knn")
predict(knn_model_RI_Ba, GlassTest)
GlassTestPredictions_RI_Ba <- predict(knn_model_RI_Ba, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Ba, GlassTest$Type)

train(Type ~ RI + Fe, data = GlassTrain, method = "knn")
knn_model_RI_Fe <- train(Type ~ RI + Fe, data = GlassTrain, method = "knn")
predict(knn_model_RI_Fe, GlassTest)
GlassTestPredictions_RI_Fe <- predict(knn_model_RI_Fe, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Fe, GlassTest$Type)

train(Type ~ RI + Mg + Na, data = GlassTrain, method = "knn")
knn_model_RI_Mg_Na<- train(Type ~ RI + Mg + Na, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_Na, GlassTest)
GlassTestPredictions_RI_Mg_Na <- predict(knn_model_RI_Mg_Na, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_Na, GlassTest$Type)

train(Type ~ RI + Mg + Al, data = GlassTrain, method = "knn")
knn_model_RI_Mg_Al <- train(Type ~ RI + Mg + Al, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_Al, GlassTest)
GlassTestPredictions_RI_Mg_Al <- predict(knn_model_RI_Mg_Al, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_Al, GlassTest$Type)

train(Type ~ RI + Mg + Si, data = GlassTrain, method = "knn")
knn_model_RI_Mg_Si <- train(Type ~ RI + Mg + Si, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_Si, GlassTest)
GlassTestPredictions_RI_Mg_Si <- predict(knn_model_RI_Mg_Si, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_Si, GlassTest$Type)

train(Type ~ RI + Mg + K, data = GlassTrain, method = "knn")
knn_model_RI_Mg_K <- train(Type ~ RI + Mg + K, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_K, GlassTest)
GlassTestPredictions_RI_Mg_K <- predict(knn_model_RI_Mg_K, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_K, GlassTest$Type)

train(Type ~ RI + Mg + Ca, data = GlassTrain, method = "knn")
knn_model_RI_Mg_Ca <- train(Type ~ RI + Mg + Ca, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_Ca, GlassTest)
GlassTestPredictions_RI_Mg_Ca <- predict(knn_model_RI_Mg_Ca, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_Ca, GlassTest$Type)

train(Type ~ RI + Mg + Ba, data = GlassTrain, method = "knn")
knn_model_RI_Mg_Ba <- train(Type ~ RI + Mg + Ba, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_Ba, GlassTest)
GlassTestPredictions_RI_Mg_Ba <- predict(knn_model_RI_Mg_Ba, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_Ba, GlassTest$Type)

train(Type ~ RI + Mg + Fe, data = GlassTrain, method = "knn")
knn_model_RI_Mg_Fe <- train(Type ~ RI + Mg + Fe, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_Fe, GlassTest)
GlassTestPredictions_RI_Mg_Fe <- predict(knn_model_RI_Mg_Fe, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_Fe, GlassTest$Type)


train(Type ~ RI + Mg + K +Al, data = GlassTrain, method = "knn")
knn_model_RI_Mg_K_Al <- train(Type ~ RI + Mg + K + Al, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_K_Al, GlassTest)
GlassTestPredictions_RI_Mg_K_Al <- predict(knn_model_RI_Mg_K_Al, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_K_Al, GlassTest$Type)


train(Type ~ RI + Mg + K + Si, data = GlassTrain, method = "knn")
knn_model_RI_Mg_K_Si <- train(Type ~ RI + Mg + K + Si, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_K_Si, GlassTest)
GlassTestPredictions_RI_Mg_K_Si <- predict(knn_model_RI_Mg_K_Si, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_K_Si, GlassTest$Type)

train(Type ~ RI + Mg + K + Ca, data = GlassTrain, method = "knn")
knn_model_RI_Mg_K_Ca <- train(Type ~ RI + Mg + K + Ca, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_K_Ca, GlassTest)
GlassTestPredictions_RI_Mg_K_Ca <- predict(knn_model_RI_Mg_K_Ca, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_K_Ca, GlassTest$Type)

train(Type ~ RI + Mg + K + Ba, data = GlassTrain, method = "knn")
knn_model_RI_Mg_K_Ba <- train(Type ~ RI + Mg + K + Ba, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_K_Ba, GlassTest)
GlassTestPredictions_RI_Mg_K_Ba <- predict(knn_model_RI_Mg_K_Ba, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_K_Ba, GlassTest$Type)

train(Type ~ RI + Mg + K + Fe, data = GlassTrain, method = "knn")
knn_model_RI_Mg_K_Fe <- train(Type ~ RI + Mg + K + Fe, data = GlassTrain, method = "knn")
predict(knn_model_RI_Mg_K_Fe, GlassTest)
GlassTestPredictions_RI_Mg_K_Fe <- predict(knn_model_RI_Mg_K_Fe, GlassTest)
confusionMatrix(GlassTestPredictions_RI_Mg_K_Fe, GlassTest$Type)
