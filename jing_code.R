
#ML FINAL
#date: 4/9
library(tidyverse)
library(caret)
library(rpart.plot)
library(glmnet)
library(Amelia)
library(pROC)
library(gbm)

###DATA CLEANING

#Load data using path of where file is stored
load("exposome.RData")

#Merge all data frames into a single data frame. FYI, this is just a shortcut by combining baseR with piping from tidyverse. There are other ways of merging across three data frames that are likely more elegant.

hw9<-merge(exposome,phenotype,by="ID") %>% merge(covariates, by="ID")

#Strip off ID Variable
hw9$ID<-NULL

#make variable as factor
hw9$hs_asthma<-as.factor(hw9$hs_asthma)
str(hw9$hs_asthma)

#Investigate whether there is imbalance of outcome
summary(hw9$hs_asthma) #There is imbalance in asthma outcome (yes:no = ~1:9 )


###DATA PARTITION
set.seed(100)
train_index<-createDataPartition(y=hw9$hs_asthma, p=0.7, list=FALSE)
hw9_train<-hw9[train_index,] #912 obs
hw9_test<-hw9[-train_index,] #389 obs






###PREDICTION VARIABLES SELECTION FROM ALL VARIABLES

#Pt. 1 - remove features with >0.8 correlation on the training data

# Finding correlated predictors
hw9_train_numeric<- hw9_train %>% dplyr::select(where(is.numeric))
hw9_train_categorical <- hw9_train %>% dplyr::select(!where(is.numeric))

correlations<-cor(hw9_train_numeric, use="complete.obs")
high.correlations<-findCorrelation(correlations, cutoff=0.8)  

# findCorrelation() searches through a correlation matrix and returns a vector of integers
# corresponding to COLUMNS to remove to reduce pair-wise correlations

# Remove highly correlated features
hw9_train_low_corr<-hw9_train_numeric[,-high.correlations] #241 variables -> 145 variables (continuous)

#Combine the low-correlated numerical variables & categorical variables
final_data = bind_cols(hw9_train_low_corr, hw9_train_categorical)

#codebook_simple = codebook %>% select(variable_name, var_type, description)
hw9_train_categorical 

#part 2 - GBM Model for features selection;

set.seed(100)

#only running a few bootstrapped samples
control.settings<-trainControl(number = 5)
gbm_hyp<-expand.grid(n.trees=(0:10)*100, shrinkage=c(0.01, 0.001), interaction.depth=c(1,3), n.minobsinnode=10)

model_gbm<-train(hs_asthma~., data=hw9_train, method="gbm", distribution="bernoulli", 
                 verbose=F, tuneGrid=gbm_hyp, trControl=control.settings)
#model了全部，data set 不对，需要更改后再run。

confusionMatrix(model_gbm) #0.8911
varImp(model_gbm) #出现前20名单 100% - 36%


#从223variables里选features后再放进去model。

### Model building






##MODEL 1: REGRESSION

#test lasso;
lambda<-10^seq(-3,3, length=100)
model_la<-train(
  hs_asthma ~., data=hw9_train, method="glmnet", family="binomial",
  trControl=trainControl("cv", number=10, sampling="down"), preProc=c("center", "scale"), tuneGrid=expand.grid(alpha=1, lambda=lambda) 
)

#test;
model_la$bestTune
model_la$results[3,]
confusionMatrix(model_la) 
plot(varImp(model_la)) 


##MODEL 2: RANDOM FOREST


##MODEL 3: UNSUPERVISED MODEL


### Compared Models


### Model evaluation
