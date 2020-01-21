library("readxl")
library("dplyr")
library(tidyr)
library("readr")

df1 <- read_excel('C:/Users/sree.vinnakoti/Downloads/SaleData.xlsx')
#dfItem <- df1(group_by(Item))
#print(df1)
#Q1
dfItem <- df1%>%group_by(Item)%>%summarise(Value=min(Sale_amt))
print(dfItem)

#Q2
df1$Year <- substr(df1$OrderDate,1,4)
dfSaleYearReg <- df1%>%group_by(Year,Region)%>%summarise(Value=sum(Sale_amt))
print(dfSaleYearReg)

#Q3
dte <- Sys.Date()
print(dte)
df1$days_diff <- difftime(dte,df1$OrderDate,units=c("days"))
print(df1)

#Q4
df4 <- df1 %>% distinct(Manager, SalesMan)
#print(df4)
output <- aggregate(df4, by=list(df4$Manager), paste, collapse=", ")
#print(output)
dfSls <- data.frame(Salesman=output$'SalesMan')
#print(dfSls)
dfMng <- data.frame(Manager=output$Group.1)
#print(dfMng)
dfMngSls <- cbind(dfMng,dfSls)
print(dfMngSls)

#Q5
df5 <- df1%>%group_by(Region)%>%summarise(Sales=sum(Sale_amt))
#print(df5)
df52 <- df1 %>% distinct(Region, SalesMan) 
#print(df52)
df53 <- data.frame(df52 %>% group_by(Region) %>% tally())
#print(df53)
dfReSlCUn <- cbind(df5,Salesmancount=df53$n)
print(dfReSlCUn)

#Q6
df6 <- data.frame(df1%>%group_by(Manager)%>%summarise(Sales=sum(Sale_amt)))
totalSale <- sum(df1$Sale_amt)
df6$Sales <- (df6$Sales/totalSale)*100
print(df6)

dfIm <- read.csv('C:/Users/sree.vinnakoti/Downloads/imdb.csv')
#print(head(dfIm))
#Q7
print(dfIm[5,]['imdbRating'])

#Q8
print(min(dfIm$duration,na.rm=TRUE))
print(max(dfIm$duration,na.rm=TRUE))

#Q9
dfi2 <- dfIm[order(dfIm[,9], -as.numeric(dfIm[,6])), ]
print(head(dfi2,20))

#Q10
dfi3 <- dfIm[((as.numeric(dfIm$ratingCount)>2000000) | (as.numeric(dfIm$ratingCount)<1000000)) & (dfIm$duration>1800 & dfIm$duration<10800),]
print(nrow(dfi3))

dfDm <- read.csv('C:/Users/sree.vinnakoti/Downloads/diamonds.csv')
print(head(dfDm,30))

#Q11
print(nrow(dfDm))
print(nrow(dfDm)-nrow(distinct(dfDm)))

#dfD2 <- dfDm %>% drop_na("cut")
#dfD3 <- dfD2 %>% drop_na("carat")
#dfDm <- dfDm[!(is.na(dfDm$cut)&is.na(dfDm$carat)),]
#dfDm <- dfDm[!(is.na(dfDm$cut)) & !(is.na(dfDm$carat)),] 
#Q12
dfDm <- na.omit(dfDm, cols = c("cut", "carat"))
print(nrow(dfDm))

#Q13
dfDn2 <- unlist(lapply(dfDm, is.numeric)) 
#print(dfDn2)
dfDnm <- dfDm[,dfDn2]
print(head(dfDnm))

#Q14
dfDm$volume <- ifelse(dfDm$depth > 60, dfDm$x*dfDm$y*as.numeric(dfDm$z), 8)
print(head(dfDm,30))

#Q15
print(sum(is.na(dfDm$price)))
dfDm$price[is.na(dfDm$price)] <- mean(dfDm$price)
print(sum(is.na(dfDm$price)))
