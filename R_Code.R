library('dplyr')
library('ggplot2')
library(rpart)
library(rpart.plot)
library('randomForest')

# Load and Check Data
train = read.csv('../input/train.csv', stringsAsFactors = F) #891 rows
test = read.csv('../input/test.csv', stringsAsFactors = F)   #418 rows

# Binding the training and test dataset, before EDA
complete_set = bind_rows(train,test)  #1309 rows
str(complete_set)

head(complete_set, nrow = 10)

## HANDLING MISSING VALUES ##

# Passengers 62 and 830 are missing Embarkment. Removing 62 and 830 Passenger IDs
embark_fare = complete_set %>% filter(PassengerId != 62 & PassengerId != 830)

# Using ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot()+
  geom_hline(aes(yintercept = 80), color = 'red', linetype = 'dashed', lwd =2)

# Since their fare was $80, they are most likely to have embarked from C
complete_set$Embarked[c(62,830)] = 'C'

# Passenger 1044 has missing Fare.
# Replacing missing fare value with median fare for class/embarkment
complete_set$Fare[1044] = median(complete_set[complete_set$Pclass == '3' & complete_set$Embarked == 'S', ]$Fare, na.rm = TRUE)

# Number of missing Age values
sum(is.na(complete_set$Age))  #263 rows

Age_unknwn = complete_set[is.na(complete_set$Age)==TRUE,-2]
Age_knwn = complete_set[is.na(complete_set$Age)==FALSE,-2]

set.seed(123)

# Build a model predicting age 
age_model <- rpart(Age~Pclass+Sex+SibSp+Parch+Fare, data = Age_knwn, method = "class", control = rpart.control(cp = 0))
plotcp(age_model)
age_model_pruned <- prune(age_model, cp = 0.0017)

Age_knwn$pred <- predict(age_model_pruned, Age_knwn, type = "class")
mean(Age_knwn$pred == Age_knwn$Age)

Age_unknwn$pred <- predict(age_model_pruned, Age_unknwn, type = "class")
complete_set$Age[is.na(complete_set$Age)==TRUE] = Age_unknwn$pred

Age_knwn$pred = as.numeric(Age_knwn$pred)

par(mfrow=c(1,2))
hist(Age_knwn$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(Age_knwn$pred, freq=F, main='Age: Predicted Data', 
     col='lightgreen', ylim=c(0,0.04))

Age_unknwn$pred <- predict(age_model_pruned, Age_unknwn, type = "class")
complete_set$Age[is.na(complete_set$Age)==TRUE] = Age_unknwn$pred

# PREDICTION
# Spliting the data back into a train set and a test set

train = complete_set[1:891,]
test  = complete_set[892:1309,]

train$Sex = as.factor(as.character(train$Sex))
train$Embarked = as.factor(as.character(train$Embarked))

rf_model <- randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data = train)

train$pred <- predict(rf_model, train)
mean(train$pred ==train$Survived)

par(mfrow=c(1,2))
hist(as.numeric(train$Survived), freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(as.numeric(train$pred), freq=F, main='Age: Predicted Output', 
     col='lightgreen', ylim=c(0,0.04))

# Show model error
plot(rf_model, ylim=c(0,0.36))
#legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

test$Sex = as.factor(as.character(test$Sex))
test$Embarked = as.factor(as.character(test$Embarked))

#rf_model <- randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data = test)

test$pred = predict(rf_model, test)
test$Survived = test$pred

solution <- data.frame(PassengerID = test$PassengerId, Survived = test$Survived)
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)


