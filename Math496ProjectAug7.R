#Author: Kristen Bystrom
#Purpose: For Math 496 Final Project, determines the activity of a human using smartphone sensors
#Last Date Modified: 2018-08-07

##############################################################
#0 SET UP ENVIRONMENT
##############################################################

  set.seed(42)
  
  #Install the packages if you have not already installed them on your computer
  #install.packages("devtools")
  #install.packages("corrplot")
  #install.packages("RandPro")
  #install.packages("dpplyr")
  #install.packages("changepoint")
  #install.packages("ecp")
  #install.packages("ggplot2")
  #install.packages("IRISSeismic")
  #install.packages("devtools")
  #install.packages('seewave')
  
  #Load Libraries
  library(devtools)
  library(corrplot)
  library(caret)
  library(RandPro)
  library(dplyr)
  library(changepoint)
  library(ecp)
  library(ggplot2)
  library(IRISSeismic)
  library(seewave)


##############################################################
#1 READ IN DATA
##############################################################
 
  test <- read.csv("test.csv")
  head(test, n= 2)
  
  train <- read.csv("train.csv")
  head(train, n =2)

##############################################################
#2 CREATE PLOTS OF CORRELATIONS AND ACTIVITIES
##############################################################

  # Check some summary plots/stats
  corTrain <- cor(train[, (3:100)])
  corrplot(corTrain, method = 'color',tl.pos = "n")
  
  plot(train$activity, las = 3, col = 'darkred')

##############################################################
#3 DEFINE RANDOM PROJECTION FUNCTION
##############################################################

  #This code block is from the clusterv source code fro random projections available at:
  #http://homes.di.unimi.it/valenti/SW/clusterv-source/rp.R
  
  # Random projections to a lower dimension subspace with the Achlioptas' projection matrix
  # The projection is performed using a  projection matrix P s.t. Prob(P[i,j]=sqrt(3))=Prob(P[i,j]=-sqrt(3)=1/6;
  # Prob(P[i,j]=0)=2/3
  # Input:
  # d : subspace dimension
  # m : data matrix (rows are features and columns are examples)
  # scaling : if TRUE (default) scaling is performed
  # Output:
  # data matrix (dimension d X ncol(m)) of the examples projected in a d-dimensional random subspace
  Achlioptas.random.projection <- function(d=2, m, scaling=TRUE){
    d.original <- nrow(m);
    if (d >= d.original)
      stop("norm.random.projection: subspace dimension must be lower than space dimension", call.=FALSE);
    # Projection matrix
    P <- floor(runif(d*d.original,1,7)); # generate a vector 1 to 6 valued
    sqr3 <- sqrt(3);
    P[P==1] <- sqr3;
    P[P==6] <- -sqr3;
    P[P==2 | P==3 | P==4 | P==5] <- 0;
    P <- matrix(P, nrow=d);
    
    # random data projection
    if (scaling == TRUE)
      reduced.m <- sqrt(1/d) * (P%*%m)
    else 
      reduced.m <- P%*%m
    reduced.m 
  }

##############################################################
#4 DETERMINE REQUIRED NUMBER OF DIMENSIONS TO SATISFY JLT
##############################################################

  # Prediction of the dimension of random projection we need to obtain a given distortion according to JL lemma
  # Input:
  # n : cardinality of the data
  # epsilon : distortion (0 < epsilon <= 0.5)
  # Output:
  # minimum dimension
  JL.predict.dim <- function(n, epsilon=0.5) {
    d <- 4 * (log(n) / epsilon^2);
    ceiling(d)
  }
  
  n = length(train$activity)-2
  JL.predict.dim(n, epsilon = 0.5) 
  #This returns 132, however we don't have the computational resources to process this
  # We will proceed with R3 instead of R128

##############################################################
#5 CHANGE POINT DETECTION USING 3 RANDOM PROJECTIONS (We lack computation time to use 128)
##############################################################

  # Remove activity and "rn" columns from matrix so we only have numeric data
  trainMatrix <- t(data.matrix(train[,-(1:2)]))
  
  # Get random projections onto R3 (d=3)
  rpResults<- data.frame(t(Achlioptas.random.projection(d = 3, trainMatrix, scaling = TRUE)))
  
  # Check random projection results
  head(rpResults)
  
  # Only run the below line of code if you have lots of time on your hands :p 
  # We have conveniently saved the results and hardcoded them in so anyone can follow along
  
  # predictedCP = e.divisive(data.matrix(rpResults), min.size = 2, sig.lvl = 0.5)
  
  #hard code results in to avoid rerunning above line code
  predictedCP = c(        1,        9,        25,        43,        52,        59,        74,        83,        101,        109,        114,        132,        141,        169,        199,        234,        257,        268,        277,        285,        294,        301,        311,        322,        331,        347,        369,        384,        396,        409,        417,        450,        459,        466,        473,        501,        514,        523,        534,        549,        565,        573,        599,        605,        614,        623,        650,        661,        668,        681,        693,        706,        715,        722,        743,        757,        769,        779,        807,        820,        828,        836,        848,        872,        880,        896,        912,        926,        935,        945,        955,        978,        989,        996,        1006,        1029,        1048,        1064,        1073,        1107,        1117,        1127,        1136,        1154,        1166,        1199,        1233,        1266,        1284,        1301,        1314,        1327,        1331,        1342,        1353,        1374,        1384,        1390,        1399,        1425,        1436,        1459,        1491,        1510,        1532,        1543,        1566,        1586,        1627,        1647,        1684,        1694,        1708,        1713,        1725,        1745,        1767,        1774,        1795,        1805,        1810,        1847,        1874,        1888,        1923,        1952,        1962,        1968,        1976,        1989,        2014,        2025,        2036,        2058,        2085,        2099,        2107,        2112,        2121,        2154,        2168,        2178,        2187,        2219,        2239,        2264,        2297,        2315,        2332,        2342,        2358,        2383,        2401,        2404,        2414,        2436,        2449,        2469,        2510,        2541,        2565,        2569,        2583,        2597,        2607,        2628,        2654,        2666,        2673,        2681,        2696,        2712,2731,        2744,        2759,        2767,        2786,        2790,        2799,        2816,        2830,        2840,        2852,        2855,        2864,        2875,        2878,        2887,        2913,        2920,        2925,        2935,        2953,        2962,        2974,        2983,        2988,        2995,        3016,        3041,        3066,        3079,        3097,        3104,        3130,        3145,        3159,        3182,        3191,        3212,        3221,        3239,        3255,        3290,        3317,        3349,        3375,        3389,        3399,        3407,        3419,        3445,        3467,        3491,        3512,        3522,        3555,        3572,        3587,3610)

##############################################################
#6 GET LIST OF ACTUAL CHANGEPOINTS USING TRAINING DATA
##############################################################

  # Get actual changepoints
  train$changepoint[1] = TRUE
  for (x in 2:length(train$activity)){
  train$changepoint[x] = (train$activity[x] != train$activity[x-1])
  }
  actualCP <- which(train$changepoint, arr.ind = TRUE)

##############################################################
#7 PLOT PREDICTED AND ACTUAL CHANGEPOINTS
##############################################################

  plot(rpResults[1:100,'X1'],type="l", col=alpha("black", 0), lty = 3, ann=FALSE,ylim=range(-1:1)) 
  abline(v = actualCP, col = "blue")
  abline(v = predictedCP, col = "red", lty = 2)


##############################################################
#8 CALCULATE SUM OF SQUARES ERROR & MEAN SQUARED ERROR
##############################################################

  #Create a matrix with the differences between each of the actual and predicted points
  i = 1
  j = 1
  accuracyCP = matrix(nrow = length(predictedCP), ncol = length(actualCP))
  for(i in 1:length(predictedCP)){ # iterate over each row
      for(j in 1:length(actualCP)){ # iterate over each column
          accuracyCP[i,j]=abs(predictedCP[i]-actualCP[j])
      }
  }
  
  #Create 2-norm function (sum of squares error function)
  norm_2 <- function(x) sqrt(sum(x^2))
  
  #Compute the sum of squares error
  print("Sum or Squares Error:")
  norm_2(apply(accuracyCP,2,min))
  
  #Compute the mean squared error
  print("Mean Squared Error:")
  norm_2(apply(accuracyCP,2,min))/length(actualCP)
  
  # Get the minimum of each column which will be the closest predicted point to each actual point
  distData = data.frame(dist = apply(accuracyCP,2,min))

##############################################################
#9 PLOT ERRORS
##############################################################

  #Plot histogram of distances to see that we are often very close or exactly correct for our changepoints except for a couple of outliers
  ggplot(data=distData, aes(distData$dist)) + 
    geom_histogram(aes(y =..density..), 
                   bins = 15,
                   col="black", 
                   fill="darkred", 
                   alpha=.8) + 
    geom_density(col="grey50", lty = 2) + 
    labs(title="", x="Distance to Closest Predicted Change Point", y="Frequency")
  
  #For each row, determine for each row what the activity number is (1 until the first changepoint, then 2, and so on)
  currentActNum = 1
  train$predActNum[1] = currentActNum
  rpResults$predActNum[1] = currentActNum
  for(i in 2:length(rpResults$X1)){
      train$predActNum[i] = currentActNum
      rpResults$predActNum[i] = currentActNum
      if(i %in% predictedCP){
          currentActNum = currentActNum + 1
      }
  }
  
  #Also, for later we will need the real activity number
  currentActNum2 = 1
  train$ActNum[1] = currentActNum2
  rpResults$ActNum[1] = currentActNum2
  for(i in 2:length(rpResults$X1)){
    train$ActNum[i] = currentActNum2
    rpResults$ActNum[i] = currentActNum2
    if(i %in% actualCP){
      currentActNum2 = currentActNum2 + 1
    }
  }
  
##############################################################
#10 GET TIME SERIES BETWEEN EACH CHANGEPOINT (predicted activities & actual activities)
##############################################################

  # The following code was modified from http://www.di.fc.ul.pt/~jpn/r/fourier/fourier.html
  # returns the x.n time series for a given time sequence (ts) and
  # a vector with the amount of frequencies k in the signal (X.k)
  get.trajectory <- function(X.k,ts,acq.freq) {
    
    N   <- length(ts)
    i   <- complex(real = 0, imaginary = 1)
    x.n <- rep(0,N)           # create vector to keep the trajectory
    ks  <- 0:(length(X.k)-1)
    
    for(n in 0:(N-1)) {       # compute each time point x_n based on freqs X.k
      x.n[n+1] <- sum(X.k * cos(2*pi*ks*n/N)) / N # change this to the cos function
    }
    
    x.n * acq.freq 
  }

##############################################################
#11 CONVERT TO FOURIER
##############################################################

  # cs is the vector of complex points to convert
  #Explore FFT Results
  convert.fft <- function(cs, sample.rate=1) {
    cs <- cs / length(cs) # normalize
  
    distance.center <- function(c)signif( Mod(c),        4)
    angle           <- function(c)signif( 180*Arg(c)/pi, 3)
    
    df <- data.frame(cycle    = 0:(length(cs)-1),
                     freq     = 0:(length(cs)-1) * sample.rate / length(cs),
                     strength = sapply(cs, distance.center),
                     delay    = sapply(cs, angle))
    df
  }
  
  #View the fourier cycles for a couple of the unidentified activities
  convert.fft(fft(rpResults[which(rpResults$predActNum == 1),"X1"]))
  convert.fft(fft(rpResults[which(rpResults$predActNum == 2),"X1"]))
  convert.fft(fft(rpResults[which(rpResults$predActNum == 4),"X1"]))
  
  X.k = fft(rpResults[which(rpResults$predActNum == 1),"X1"])
  X.k = fft(X.k, inverse = TRUE)/length(X.k)
  time     <- 2.56                            # measuring time interval (seconds) from data description
  acq.freq <- 50                          # data acquisition frequency (Hz) from data description
  ts  <- seq(0,time-1/acq.freq,1/acq.freq) # vector of sampling time-points (s) 
  x.n <- get.trajectory(X.k,ts,acq.freq)   # create time wave
  
##############################################################
#12 DEFINE FUNCTION TO PLOT HARMONIC SERIES
##############################################################

  #Function to plot a harmonic series
  #Modified from http://www.di.fc.ul.pt/~jpn/r/fourier/fourier.html
  plot.harmonic <- function(Xk, i, ts, acq.freq, color="red") {
    Xk.h <- rep(0,length(Xk))
    Xk.h[i+1] <- Xk[i+1] # i-th harmonic
    harmonic.trajectory <- get.trajectory(Xk.h, ts, acq.freq=acq.freq)
    points(ts, harmonic.trajectory, type="l", col=color)
  }
  
  #Plot the inverse fast fourier transform with its first 3 harmonic oscillations
  #This will remove the imaginary parts which is ok because the imaginary parts are all 0.
  
  plot(ts,x.n,type="l",ylim=c(-5,4),lwd=2, xlab = "Time", ylab = "Position")
  abline(v=0:time,h=-2:4,lty=3); abline(h=0)
  plot.harmonic(X.k,1,ts,acq.freq,"red")
  plot.harmonic(X.k,2,ts,acq.freq,"green")
  plot.harmonic(X.k,3,ts,acq.freq,"blue")
  

##############################################################
#13 PLOT FOURIER TRANSFORM EXAMPLES
##############################################################
  #Find table position with max length
  a = table(rpResults$predActNum)
  which(a==max(a))
  #Find max length
  max(a)
  
  # Plot Fourier transform for max length activity
  X.k.max = fft(rpResults[which(rpResults$predActNum == 158),"X1"])
  X.k.max = fft(X.k.max, inverse = TRUE)/length(X.k.max)
  time     <- 2.56                            # measuring time interval (seconds) from data description
  acq.freq <- 50                          # data acquisition frequency (Hz) from data description
  ts  <- seq(0,time-1/acq.freq,1/acq.freq) # vector of sampling time-points (s) 
  
  x.n.max <- get.trajectory(X.k.max,ts,acq.freq)   # create time wave
  
  plot(ts,x.n.max,type="l",ylim=c(-8,8),lwd=2, main= "", xlab = "Time", ylab = "Position")
  abline(v=0:time,h=-2:4,lty=3); abline(h=0)
  
  plot.harmonic(X.k.max,1,ts,acq.freq,"red")
  plot.harmonic(X.k.max,2,ts,acq.freq,"green")
  plot.harmonic(X.k.max,3,ts,acq.freq,"blue")
  length(x.n.max)

##############################################################
#14 CREATE MATRX OF ACTIVITIES WITH THE FOURIER TRANFORMS OF THEIR RANDOM PROJECTIONS (PREDICTED & ACTUAL)
##############################################################

  time     <- 2.56                         # measuring time interval (seconds) from data description
  acq.freq <- 50                           # data acquisition frequency (Hz) from data description
  ts  <- seq(0,time-1/acq.freq,1/acq.freq) # vector of sampling time-points (s) 
  randPros = c("X1", "X2", "X3")
  
  
  ##############################################################
  ##############################################################
  ##############################################################
  #Actual:

    #Get the three waves for each activity
    #Create Matrix with one activity segment per row, one series per column. 
    #This should give an 398*3*128 matrix where 398 is the number of distinct activities
    activityMatrixActual = array(rep(0, length(unique(rpResults$ActNum))* 3*128), dim = c( 128, 3, length(unique(rpResults$ActNum))))
    dim(activityMatrixActual)
    
    # This loop creates the time waves for each activities' three random projections and stores all this data in a matrix (predited activities)
    for(i in 1:3){
      for(j in 1:398){
        X.k.j2 = fft(rpResults[which(rpResults$ActNum == j),randPros[i]])
        X.k.j2 = fft(X.k.j2, inverse = TRUE)/length(X.k.j2)
        x.n.j2 <- get.trajectory(X.k.j2,ts,acq.freq)   # create time wave
        activityMatrixActual[,i,j] = as.numeric(x.n.j2)
      }
    }
    dim(activityMatrixActual)
    # 128 Fourier Coefficients
    # 3 Random Projections
    # 398 Unidentified Activities 
    
  ##############################################################
  ##############################################################
  ##############################################################
  #Predicted:

    #Get the three waves for each activity
    #Create Matrix with one activity segment per row, one series per column. 
    #This should give an 229*3*128 matrix where n is the number of distinct activities
    activityMatrix = array(rep(0, length(unique(rpResults$predActNum))* 3*128), dim = c( 128, 3, length(unique(rpResults$predActNum))))
    dim(activityMatrix)
    
    # This loop creates the time waves for each activities' three random projections and stores all this data in a matrix (predited activities)
    for(i in 1:3){
        for(j in 1:229){
            X.k.j = fft(rpResults[which(rpResults$predActNum == j),randPros[i]])
            X.k.j = fft(X.k.j, inverse = TRUE)/length(X.k.j)
            x.n.j <- get.trajectory(X.k.j,ts,acq.freq)   # create time wave
            activityMatrix[,i,j] = as.numeric(x.n.j)
        }
    }
    dim(activityMatrix)
    # 128 Fourier Coefficients
    # 3 Random Projections
    # 229 Unidentified Activities 
    
##############################################################
#15 CHECK IF WE CAN USE SPECTRAL COHERENCE (spoiler - the answer is no!)
##############################################################

  #Merge two time series together
  ts1 <- ts(activityMatrix[,1,1],frequency=50)
  ts2 <- ts(activityMatrixActual[,1,5],frequency=50)
  x <- ts.union(ts1,ts2)
  DF = crossSpectrum(x)

  # Calculate the transfer function
  transferFunction <- DF$Pxy / DF$Pxx
  transferAmp <- Mod(transferFunction)
  transferPhase <- pracma::mod(Arg(transferFunction) * 180/pi,360)
  
  #Plot Spectral Coherence
  dev.off()
  # Plot
  plot(1/DF$freq,transferAmp,type='l',log='x',
       xlab="Period (sec)",
       main="Transfer Function Amplitude")
  plot(1/DF$freq,transferPhase,type='l',log='x',
       xlab="Period (sec)", ylab="degrees",
       main="Transfer Function Phase")
  #We stopped pursuing spectral coherence at this point due to coding problems

##############################################################
#16 SWITCH TO CROSS CORRELATION FUNCTION INSTEAD OF SPECTRAL COHERENCE (TAKES ABOUT 20 minutes to run)
##############################################################

  #Compute mean ccf for all pairs of activities using a somewhat undesirable nested for loop situation
  #Rows are predicted events, columns are actual events, values are "closeness" in terms of CCF, where higher values mean more close
  allccf <- matrix(nrow = 229, ncol=398)
  for(k in 1:229){
    for(l in 1:398){
      maxCCF =rep(0, n = 9)
      m = 1
      for(i in 1:3){ #index of random projection for activity 1
        for(j in 1:3){ #index of random projection for acitivity 2
          print(paste("k is: ", k, "l is: ", l,"i is: ", i, ", j is: ", j, ", m is: ", m))
          ts1 <- ts(activityMatrix[,i,k],frequency=50)
          ts2 <- ts(activityMatrixActual[,j,l],frequency=50)
          maxCCF[m]=max(abs((ccf(ts1,ts2, plot = FALSE))$acf))
          m = m+1
        }
      }
      allccf[k,l] = mean(maxCCF)
    }
  }
  allccf
  write.csv(allccf, "allccf_output.csv")

##############################################################
#17 IMPLEMENT K-NEAREST NEIGHBORS
##############################################################

  #install.packages("FastKNN)
  library(FastKNN)
  
  #Get matrix of 3 closest neighbours
  n = 229 #(number of rows to predict)
  k=5 #THis is a fairly arbitrary choice, as parameter tuning is outside the scope of this project
  nn = matrix(0,n,k) # n x k
  for (i in 1:n)
    nn[i,] = k.nearest.neighbors(i, 1/allccf, k = k)
  
  #Write function that gets most common value
  MaxTable <- function(InVec, mult = FALSE) {
    if (!is.factor(InVec)) InVec <- factor(InVec)
    A <- tabulate(InVec)
    if (isTRUE(mult)) {
      levels(InVec)[A == max(A)]
    } 
    else levels(InVec)[which.max(A)]
  }
  
  #Get event prediction for each activity
  finalPrediction = rep("to be filled", times= 229)
  for (i in 1:n){
    acts = rep("to be filled", times = 3)
    for (j in 1:3){
      acts[j] = as.character(train[which(train$ActNum == nn[i,j]),c("activity")][1])
      
    }
    finalPrediction[i] = MaxTable(factor(acts))
  }
  finalPrediction #this is event prediction for each activity 

##############################################################
#18 COMPUTE ACCURACY
##############################################################
  #We actually can't predict accuracy for our test data because it is an open kaggle competition. 
  #However, below is our code that tests accuracy on the training data
  
  #Need to get predictions for each event into rows
  getPred <- function(activityNum){
    activityName = finalPrediction[activityNum]
    return(activityName)
  }
  
  train$predActivity= lapply(train$predActNum,getPred)
  
  accuracy = function(actual, predicted) {
    mean(actual == predicted)
  }
  
  accuracy(train$activity, train$predActivity) #Our accuracy is 21%
##############################################################
# END OF CODE
##############################################################