setwd("F:/Text/College/University of Wisconsin/740 Data Mining - Fall 2018/FinalProject")
cust<-read.csv("Customer_Sales_Transactional_data.csv")
summary(cust)
dim(cust) #91698    36

#converting to factors
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

#remove Customer_ID as well as  Customer_Value and Visitors_Type, as the same info is in the vars "High_value",
#"Low_value", "Regular" and "Rare_Visitors", "Frequent_Visitors", "Regular_Visitors" 
cust<-cust[,-c(1,29,33)]
names(cust)
  # [1] "Total_Sale"            "STD_Sales"             "Hist_Visits"           "W1_Min_Sale"          
  #[5] "W1_STD_Sales"          "W1_Visits"             "W2_Min_Sale"           "W2_STD_Sales"         
  #[9] "W2_Visits"             "W3_Sale"               "W3_Max_Sale"           "W3_Min_Sale"          
  #[13] "W3_STD_Sales"          "W3_Visits"             "W4_Sale"               "W4_Min_Sale"          
  #[17] "W4_STD_Sales"          "W4_Visits"             "W5_STD_Sales"          "W5_Visits"            
  #[21] "week_1"                "week_2"                "week_3"                "week_4"               
  #[25] "APV"                   "Days_since_last_visit" "CHURN"                 "High_value"           
  #[29] "Low_value"             "Regular"               "Rare_Visitors"         "Frequent_Visitors"    
  #[33] "Regular_Visitors" 


cust.scale<-data.frame(CHURN=cust$CHURN,scale(cust[,c(1:20,25,26)]),cust[,21:24],cust[,28:33])
head(cust.scale)
names(cust.scale)
######### ######### ######### ######### ######### ######### ######### #########  #########
######### classification of CHURN=0 or CHURN=1                                   ######### 
######### LDA/QDA assumes normality                                              ######### 
######### KNN performs better than logistic when relationship highly non-linear  ######### 
######### ######### ######### ######### ######### ######### ######### #########  #########

#testing correlations
corTestDF1<-cust[,1:8]
cor(corTestDF1)
corTestDF2<-cust[,7:12]
cor(corTestDF2)
corTestDF3<-cust[,14:20]
cor(corTestDF3)
  #significant correlations between predictor vars, such as:
  #Hist_Visits/Total_Sale, W3_Sale/W3_Max_Sale,W3_Min_Sale, W4_Sale/W4_Min_Sale

######### ######### ######### ######### ######### ######### ######### ######### ######  #########
######### classification of CHURN=0 or CHURN=1                                          ######### 
######### Bagging will have high variance because of high correlations                  ######### 
######### Use random forest because of correlations in predictors.  Use CV to choose P  ######### 
######### ######### ######### ######### ######### ######### ######### ######### ######  #########
######### ANN best for large data sets with nonlinear relationships. Use CV to tune     #########
######### ######### ######### ######### ######### ######### ######### ######### ######  #########

#######################################
###### Use random forest and ANN ######
#######################################

##################################################################################################
#################### Random Forest for sample of 1000 obs ########################################
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

rf.mse<-apply(group.error, 1, mean)
# [1] 0.2452472 0.2362068 0.2362276 0.2392480 0.2442478 0.2522983 0.2503187 0.2402981 0.2452882 0.2562783 0.2532783
# [12] 0.2412682 0.2552686 0.2432583 0.2522785 0.2472583 0.2522682 0.2442583 0.2492985 0.2442781 0.2452882 0.2523187
# [23] 0.2462882 0.2432884 0.2432682 0.2483183 0.2462985 0.2452682 0.2473084 0.2482981 0.2422680 0.2432583
  #mtry=2,3,4 are the lowest, though no size is very different. Will go with 2 for simplicity

###################### Final Random Forest Model and Error Rate, for mtry=2 ##############################
set.seed(22)
mtry2.rf.model<-randomForest(cust.sample$CHURN~., data=cust.sample, mtry=2, importance=T)
cust.rf.conf<-mtry2.rf.model$confusion
mtry2.rf.er<-(cust.rf.conf[1,2]+cust.rf.conf[2,1])/(sum(cust.rf.conf[,1])+sum(cust.rf.conf[,2])) #0.235

plot(final.rf.model,xlim=c(0,150))
  #Plot shows same error rate at 60 trees as at 500 trees. Can simplify for full model
##########################################################################################################


##################################################################################################
#################### Random Forest for full data set      ########################################
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
    cust.rf<-randomForest(cust$CHURN[!groupi]~., data=cust[!groupi,], mtry=j, importance=T)
    cust.rf.pred<-predict(cust.rf,newdata=cust[groupi,])
    cust.rf.conf<-table(cust.rf.pred, cust$CHURN[groupi])
    group.error[j, i]<-(cust.rf.conf[1,2]+cust.rf.conf[2,1])/(sum(cust.rf.conf[,1])+sum(cust.rf.conf[,2]))
  } #end iterate over model size
} #end iterate over folds

rf.mse<-apply(group.error, 1, mean)

