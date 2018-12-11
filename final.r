##################################################################################################
###### Bradford Simkins                                                                     ######
###### DS740: Data Mining - Final Project                                                   ######
###### 12/11/2018                                                                           ######
##################################################################################################

setwd("F:/Text/College/University of Wisconsin/740 Data Mining - Fall 2018/FinalProject")
cust<-read.csv("Customer_Sales_Transactional_data.csv")
##################################################################################################
##                      Exploring the Data                                                      ##
##################################################################################################
summary(cust)
dim(cust) #91698    36

#converting 1/0 variables to factors
cust$week_1<-as.factor(cust$week_1)
cust$week_2<-as.factor(cust$week_2)
cust$week_3<-as.factor(cust$week_3)
cust$week_4<-as.factor(cust$week_4)
cust$CHURN<-as.factor(cust$CHURN)
cust$High_value<-as.factor(cust$High_value)
cust$Low_value<-as.factor(cust$Low_value)
cust$Regular<-as.factor(cust$Regular)
cust$Rare_Visitors<-as.factor(cust$Rare_Visitors)
cust$Frequent_Visitors<-as.factor(cust$Frequent_Visitors)
cust$Regular_Visitors<-as.factor(cust$Regular_Visitors)

#checking distributions of variables
par(mfrow=c(3,3))
hist(cust[,2])
hist(cust[,3])
hist(cust[,4])
hist(cust[,5])
hist(cust[,6])
hist(cust[,7])
hist(cust[,8])
hist(cust[,9])
hist(cust[,10])

par(mfrow=c(3,3))
hist(cust[,11])
hist(cust[,12])
hist(cust[,13])
hist(cust[,14])
hist(cust[,15])
hist(cust[,16])
hist(cust[,17])
hist(cust[,18])
hist(cust[,19])

par(mfrow=c(2,2))
hist(cust[,20])
hist(cust[,21])
hist(cust[,26])
hist(cust[,27])
  #very few are normally distributed

names(cust)
  # [1] "CUSTOMER_ID"           "Total_Sale"            "STD_Sales"             "Hist_Visits"          
  #[5] "W1_Min_Sale"           "W1_STD_Sales"          "W1_Visits"             "W2_Min_Sale"          
  #[9] "W2_STD_Sales"          "W2_Visits"             "W3_Sale"               "W3_Max_Sale"          
  #[13] "W3_Min_Sale"           "W3_STD_Sales"          "W3_Visits"             "W4_Sale"              
  #[17] "W4_Min_Sale"           "W4_STD_Sales"          "W4_Visits"             "W5_STD_Sales"         
  #[21] "W5_Visits"             "week_1"                "week_2"                "week_3"               
  #[25] "week_4"                "APV"                   "Days_since_last_visit" "CHURN"                
  #[29] "Customer_Value"        "High_value"            "Low_value"             "Regular"              
  #[33] "Visitors_Type"         "Rare_Visitors"         "Frequent_Visitors"     "Regular_Visitors" 

#removing Customer_ID as well as  Customer_Value and Visitors_Type, as the same info is in the vars "High_value",
#"Low_value", "Regular" and "Rare_Visitors", "Frequent_Visitors", "Regular_Visitors" 
cust<-cust[,-c(1,29,33)]


######### ######### ######### # Notes on Methods # ######### ######### ######### ######### #########
######### Ploblem: Classification of CHURN=0 or CHURN=1                                    ######### 
######### LDA/QDA assumes normality, so would not work without significant transformations ######### 
######### KNN performs better than logistic regression when relationship highly non-linear ######### 
######### ######### ######### ######### ######### ######### ######### #########  ######### #########

#testing correlations
corTestDF1<-cust[,1:8]
cor(corTestDF1)
corTestDF2<-cust[,7:12]
cor(corTestDF2)
corTestDF3<-cust[,14:20]
cor(corTestDF3)
  #significant correlations between predictor vars, such as:
  #Hist_Visits/Total_Sale, W3_Sale/W3_Max_Sale,W3_Min_Sale, W4_Sale/W4_Min_Sale

######### ######### ######### # Notes on Methods # ######## ######### ######### ###### ###### ######
######### Bagging will have high variance because of high correlations                        ###### 
######### *** Use random forest because of correlations in predictors.  Use CV to choose P    ######
######### ######### ######### ######### ######### ######### ######### ######### ###### ###### ######
######### *** ANN best for large data sets with nonlinear relationships. Use CV to tune       ######
######### ######### ######### ######### ######### ######### ######### ######### ###### ###### ######


##################################################################################################
##                      Random Forest for sample of 1000 obs                                    ##
##################################################################################################

#get a random, but representative, subset of data to work with for testing, as 90,000+ obs runs 
#too slowly
set.seed(24)
cust.sample<-cust[sample(nrow(cust),1000),]

churn0<-length(which(cust$CHURN==0))
churn0/length(cust$CHURN) #CHURN==0 represents 67% of the original data set

churn0.sample<-length(which(cust.sample$CHURN==0))
churn0.sample/length(cust.sample$CHURN) #CHURN==0 represents 66% of the sampled data set
  #proportion of sample response variable similar to original

#random forest with cross-validation for size selection
library(randomForest)
n<-dim(cust.sample)[1]
k<-10
groups<-c(rep(1:k,floor(n/k)),1:(n-floor(n/k)*k)) #list of group labels
set.seed(22)
cvgroups<-sample(groups,n)
group.error<-matrix(,nr=32, nc=k) #matrix to hold error rates for each model size, in each fold

#10-fold cross validation
for (i in 1:k) { #iterate over folds
  groupi<-(cvgroups==i)
  for(j in 1:32){ #iterate over model size
    cust.rf<-randomForest(cust.sample$CHURN[!groupi]~., data=cust.sample[!groupi,], mtry=j, importance=T)
    cust.rf.pred<-predict(cust.rf,newdata=cust.sample[groupi,])
    cust.rf.conf<-table(cust.rf.pred, cust.sample$CHURN[groupi])
    group.error[j, i]<-(cust.rf.conf[1,2]+cust.rf.conf[2,1])/(sum(cust.rf.conf[,1])+sum(cust.rf.conf[,2]))
  } #end iterate over model size
} #end iterate over folds

rf.mse.sample<-apply(group.error, 1, mean)
  # [1] 0.2452472 0.2362068 0.2362276 0.2392480 0.2442478 0.2522983 0.2503187 0.2402981 0.2452882 0.2562783 0.2532783
  # [12] 0.2412682 0.2552686 0.2432583 0.2522785 0.2472583 0.2522682 0.2442583 0.2492985 0.2442781 0.2452882 0.2523187
  # [23] 0.2462882 0.2432884 0.2432682 0.2483183 0.2462985 0.2452682 0.2473084 0.2482981 0.2422680 0.2432583
  #mtry=2,3,4 are the lowest, though no size is very different. Will go with 2 for simplicity

## Final Random Forest Model and Error Rate, for mtry=2 with 1000 obs ##
set.seed(22)
sample.rf.model<-randomForest(cust.sample$CHURN~., data=cust.sample, mtry=2, importance=T)
cust.rf.conf<-sample.rf.model$confusion
sample.rf.er<-(cust.rf.conf[1,2]+cust.rf.conf[2,1])/(sum(cust.rf.conf[,1])+sum(cust.rf.conf[,2])) #0.235

plot(sample.rf.model,xlim=c(0,150))
  #Plot shows same error rate at 60 trees and 500 trees. Can simplify number of trees for full model
##################################################################################################
##                     End :: Random Forest for sample of 1000 obs                              ##
##################################################################################################
                            
##################################################################################################
##                        Random Forest for full data set                                       ##
##################################################################################################
n<-dim(cust)[1]
k<-10
groups<-c(rep(1:k,floor(n/k)),1:(n-floor(n/k)*k)) #list of group labels
set.seed(22)
cvgroups<-sample(groups,n)
group.error<-matrix(,nr=32, nc=k) #matrix to hold error rates for each model size, in each fold

#10-fold cross validation
for (i in 1:k) { #iterate over folds
  groupi<-(cvgroups==i)
  for(j in 1:32){ #iterate over model size
    cust.rf<-randomForest(cust$CHURN[!groupi]~., data=cust[!groupi,], mtry=j, ntree=60, importance=T)
    cust.rf.pred<-predict(cust.rf,newdata=cust[groupi,])
    cust.rf.conf<-table(cust.rf.pred, cust$CHURN[groupi])
    group.error[j, i]<-(cust.rf.conf[1,2]+cust.rf.conf[2,1])/(sum(cust.rf.conf[,1])+sum(cust.rf.conf[,2]))
  } #end iterate over model size
} #end iterate over folds

rf.er<-apply(group.error, 1, mean)

#plot of Error Rate for all mtry sizes
plot(rf.er, xlab="Size", ylab="Erorr", main="Full Model Error Rates")
  #error rates are similar to sample model error rates, but show a much clearer pattern
  #lowest error rate for size of 3 with 60 trees

## Final Random Forest Model and Error Rate, for mtry=3 with 60 trees ##
set.seed(22)
final.rf.model<-randomForest(cust$CHURN~., data=cust, mtry=3, ntree=60, importance=T)
final.rf.conf<-final.rf.model$confusion
final.rf.er<-(final.rf.conf[1,2]+final.rf.conf[2,1])/(sum(final.rf.conf[,1])+sum(final.rf.conf[,2])) #0.23127

plot(final.rf.model)
  #Final error rate for this model 0.23127
##################################################################################################
##                            End :: Random Forest for full data set                            ##
##################################################################################################


##################################################################################################
##                  Artificial Neural Network for sample of 1000 obs                            ##
##                         Tuning Number of Hidden Nodes                                        ##
##################################################################################################
library(nnet)
library(NeuralNetTools)

#grouping columns by numeric and catagorical variables
cust.sample.col.sort<-data.frame(cust.sample[,1:20], cust.sample[,25:26], cust.sample[,21:24], cust.sample[,27:33])
names(cust.sample.col.sort)

## using 10-fold cross-validation to select number of hidden nodes ##
n<-dim(cust.sample)[1] #1000
k<-10 # using 10-fold cross-validation
groups<-c(rep(1:k,floor(n/k)))
sizes<-1:20 #numbers of hidden nodes to try
misclassError<-matrix(, nr=k, nc=length(sizes) ) #fill 10x20 matrix with NAs
conv<-matrix(, nr=k, nc=length(sizes) )  #fill 10x20 matrix with NAs
set.seed(13)
cvgroups<-sample(groups,n)

#run 10-fold cross-validation
for (i in 1:k){ #iterate over folds
  groupi<-(cvgroups==i)
  #standardize sampled data
  cust.sample.train.std<-scale(cust.sample.col.sort[,1:22])
  cust.sample.train<-data.frame(cust.sample.train.std[!groupi,1:22],cust.sample.col.sort[!groupi,23:33])
  cust.sample.valid<-data.frame(scale(cust.sample.col.sort[groupi,1:22],center=attr(cust.sample.train.std,"scaled:center"),
                                      scale=attr(cust.sample.train.std,"scaled:scale")),cust.sample.col.sort[groupi,23:33])
  
  #iterate over numbers of hidden nodes
  for (j in 1:length(sizes)){
    cust.sample.fit<-nnet(CHURN~., data=cust.sample.train, size=sizes[j], trace=F, maxit=5000) 
    predictions<-predict(cust.sample.fit, cust.sample.valid, type="class")
    misclassError[i, j]<-length(which(predictions!=cust.sample.valid[, 27]))/length(predictions)
    conv[i, j]<-cust.sample.fit$convergence
  } # end iteration over numbers of hidden nodes
} # end iteration over folds

## Convergence Assessment ##
#with maxit=1000
colSums(conv) #0 0 0 0 0 1 1 0 0 1 2 4 3 3 2 7 8 8 6 5
length(which(colSums(conv)!=0)) #13
  #13 not=0, so not many converged

#trying with maxit=3000
colSums(conv) #0 0 0 0 0 0 0 0 0 0 0 2 1 0 1 2 2 1 4 0
length(which(colSums(conv)!=0)) #7
  #7 not=0, so not all converged

#trying with maxit=5000
colSums(conv) #0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  #all=0, so all converged

#average misclassification error across all folds of each number of hidden nodes
error<-apply(misclassError, 2, mean)

#plot of error rate vs. the number of hidden nodes
plot(sizes, error, type="l", main="ANN: Sample of 1000", xlab="Hidden Nodes", ylab="Error Rate")
which(error==min(error)) #5
  #while 5 hidden nodes minimizes the error rate, anything under 9 is about the same
  #probably go with 2, in this sample, but will try full data set with 2:8 hidden nodes
  #this will hopefully smooth out the curve so we can pick a more accurate number of hidden nodes

##################################################################################################
##                      Artificial Neural Network with Full Data Set                            ##
##                            Tuning Number of Hidden Nodes                                     ##
##################################################################################################

#grouping by numeric and catagorical variables
cust.col.sort<-data.frame(cust[,1:20], cust[,25:26], cust[,21:24], cust[,27:33])
names(cust.col.sort)

########### using 10-fold cross-validation to select number of hidden nodes ######################
n<-dim(cust)[1] #91698
k<-10 # using 10-fold cross-validation
groups<-c(rep(1:k,floor(n/k)),rep(1:(n-(k*floor(n/k))),1))
sizes<-2:8 #numbers of hidden nodes to try
misclassError<-matrix(, nr=k, nc=length(sizes) ) #fill 10x20 matrix with NAs
conv<-matrix(, nr=k, nc=length(sizes) )  #fill 10x20 matrix with NAs
set.seed(13)
cvgroups<-sample(groups,n)

#run 10-fold cross-validation
for (i in 1:k){ #iterate over folds
  groupi<-(cvgroups==i)
  #standardize sampled data
  cust.train.std<-scale(cust.col.sort[,1:22])
  cust.train<-data.frame(cust.train.std[!groupi,1:22],cust.col.sort[!groupi,23:33])
  cust.valid<-data.frame(scale(cust.col.sort[groupi,1:22],center=attr(cust.train.std,"scaled:center"),
                                      scale=attr(cust.train.std,"scaled:scale")),cust.col.sort[groupi,23:33])
  
  #iterate over numbers of hidden nodes
  for (j in 1:length(sizes)){
    cust.fit<-nnet(CHURN~., data=cust.train, size=sizes[j], trace=F, maxit=3000) 
    predictions<-predict(cust.fit, cust.valid, type="class")
    misclassError[i, j]<-length(which(predictions!=cust.valid[, 27]))/length(predictions)
    conv[i, j]<-cust.fit$convergence
  } # end iteration over numbers of hidden nodes
} # end iteration over folds

#with maxit=3000
colSums(conv) #0 0 0 0 0 0 0
  #converged

#average misclassification error across all folds of each number of hidden nodes
error<-apply(misclassError, 2, mean)

#plot of error rate vs. the number of hidden nodes
plot(sizes, error, type="l", main="ANN: Full Data Set", xlab="Hidden Nodes", ylab="Error Rate")
which(error==min(error)) #8
  #while 8 hidden nodes minimizes the error rate, everything between 3 and 8 is pretty close, 
  #I'm going with 3, for the sake of simplicity

##################################################################################################
##                    Artificial Neural Network for Full Data Set                               ##
##                      Tuning Number of Weight Decay Parameter                                 ##
##################################################################################################
n<-dim(cust)[1] #91698
k<-10 # using 10-fold cross-validation
groups<-c(rep(1:k,floor(n/k)),rep(1:(n-(k*floor(n/k))),1))
decayRate<-seq(.5,2,by=.1) #decay rates, random tests show now bennefit below .5 or above 2
misclassError<-matrix(, nr=k, nc=length(decayRate) ) #fill 10x16 matrix with NAs
conv<-matrix(, nr=k, nc=length(decayRate) )  #fill 10x16 matrix with NAs
set.seed(13)
cvgroups<-sample(groups,n)

#run 10-fold cross-validation
for (i in 1:k){ #iterate over folds
  groupi<-(cvgroups==i)
  #standardize sampled data
  cust.train.std<-scale(cust.col.sort[,1:22])
  cust.train<-data.frame(cust.train.std[!groupi,1:22],cust.col.sort[!groupi,23:33])
  cust.valid<-data.frame(scale(cust.col.sort[groupi,1:22],center=attr(cust.train.std,"scaled:center"),
                                      scale=attr(cust.train.std,"scaled:scale")),cust.col.sort[groupi,23:33])
  
  #iterate over numbers of hidden nodes
  for (j in 1:length(decayRate)){
    cust.fit<-nnet(CHURN~., data=cust.train, size=3, decay=decayRate[j], trace=F, maxit=3000) 
    predictions<-predict(cust.fit, cust.valid, type="class")
    misclassError[i, j]<-length(which(predictions!=cust.valid[, 27]))/length(predictions)
    conv[i, j]<-cust.fit$convergence
  } # end iteration over numbers of hidden nodes
} # end iteration over folds

#with maxit=3000, size=3
colSums(conv) #0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  # all converged

#average misclassification error across all folds of each number of hidden nodes
error<-apply(misclassError, 2, mean)

plot(decayRate, error, type="l", main="ANN: Full Data Set", xlab="Decay Rate", ylab="Error Rate")
decayRate[which(error==min(error))] #1.8
min.error<-min(error)
  #a decay rate of 1.8 minimizes the error for ANN with 3 hidden nodes, but the plot shows
  #a lot of up and down between error of .230 and .2325, so there is not a lot of difference
  #lowest error rate is 0.2307684, barely lower than random forest's of 0.23127

###########################################################################################
##                   Testing ANN with selected variables                                 ##
###########################################################################################
ga<-garson(ANN.model)
ga$data #garson shows the 6 most important predictors are
        #Days_since_last_visit, Total_Sale, W2_Min_Sale, W2_Visits, APV,W3_Sale, week_2, week_1
cust.small<-data.frame(cust$CHURN,cust$Days_since_last_visit,cust$Total_Sale,cust$W2_Min_Sale,cust$W2_Visits,
                       cust$APV,cust$W3_Sale,cust$week_2,cust$week_1)
colnames(cust.small)<-c("CHURN","Days_since_last_visit","Total_Sale","W2_Min_Sale","W2_Visits","APV","W3_Sale",
                        "week_21","week_11")

set.seed(13)
train<-sample(1:91698,60520,replace=F)

ANN.modelB<-nnet(CHURN~., data=cust.small[train,], size=3, decay=1.8, trace=F, maxit=3000)
custClass<-predict(ANN.modelB, cust.small[-train,],type="class")

#confusion matrix
table(custClass, cust.small[-train,]$CHURN)
  #custClass     0     1
  #0 17919  4443
  #1  2862  5954
  #(4443+2862)/(4443+2862+17919+5954)=0.2342998


cust.small.4<-data.frame(cust$CHURN,cust$Days_since_last_visit,cust$Total_Sale,cust$APV,cust$week_2)
colnames(cust.small.4)<-c("CHURN","Days_since_last_visit","Total_Sale","APV","week_2")


ANN.modelC<-nnet(CHURN~., data=cust.small.4[train,], size=3, decay=1.8, trace=F, maxit=3000)
custClass2<-predict(ANN.modelC, cust.small.4[-train,],type="class")

#confusion matrix
table(custClass2, cust.small.4[-train,]$CHURN)
  #custClass2     0     1
  #0 17955  4517
  #1  2826  5880
  #(4517+2826)/(4517+2826+17955+5880)=0.2355186

#using all 32 predictors for ANN results in a lowest error rate of 0.2307684
#using only the 4 predictors Days_since_last_visit, Total_Sale, APV, week_2 results 
#in an error rate of 0.2355186. This is an insignificant difference of 0.005.  It would 
#pay off in processing time to use the 4 variable model. Ideally I would tune the size
#and decay parameters again, but I used up days of time running the full data set multiple times

#############################################################################################
##                               Final ANN Model                                           ##
#############################################################################################
final.ANN.model<-nnet(CHURN~., data=cust.small.4, size=3, decay=1.8, trace=F, maxit=3000)
#############################################################################################

#plot the ANN
library(NeuralNetTools)
plotnet(final.ANN.model)
