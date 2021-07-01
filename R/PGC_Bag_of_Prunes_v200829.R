###############################################################################
###############################################################################
######## "To Bag is to Prune" Codes by Philippe GC, 2020/08/29 ################
###############################################################################
###############################################################################

#utilities
#library(gbm)
#library(earth)

r.squared = function(pred, actual) {
  rss <- sum((actual-pred)^2)
  tss <- sum((actual-mean(actual))^2)
  rsq=1-rss/tss
  #if(rsq< -0.5){rsq=-0.5}
  return(rsq)
}

###############################################################################
###############################################################################
###############################################################################
####################### Booging ###############################################
###############################################################################
###############################################################################
###############################################################################



Booging = function(y,X,X.new,B=100,mtry=0.8,sampling.rate=.75,data.aug=FALSE,noise.level=0.3,shuffle.rate=0.2,fix.seeds=TRUE,
                   bf=.5,n.trees=1000,tree.depth=3,nu=.3){
  #Row 1 are options related to Booging "ensembling"
  #Row 2 are gbm's options
  #Keep default values unless you know what you're doing
  X=as.data.frame(X)
  X.new=as.data.frame(X.new)

  #This block finds out which variables are continous
  ids=rep(NA,ncol(X))
  for(kk in 1:ncol(X)){ids[kk]=length(unique(X[,kk]))==2}
  varclass=which(!ids)

  #This scales the continous variables.
  #This is necessary because we may be adding noise later, which variance is heuristically "tuned" according to X_k ~N(0,1).
  Xall = scale(rbind(X[,varclass],X.new[,varclass]))
  X.new[,varclass] = Xall[(nrow(X)+1):nrow(Xall),]
  X[,varclass] = Xall[1:nrow(X),]

  #this prevents a GBM error when bag.frac is set too low with small samples.
  if(length(y)<100){bf=max(.4,bf)}

  ##################################################################
  ##################################################################
  # This block implements data augmentation as described in Appendix A.2
  if(data.aug){
    if(fix.seeds){set.seed(1)}
    #creates too copies of X
    X.da.in1 = X
    X.da.in2 = X

    #find continous variables
    varclass=which(!ids)

    #gaussian noise infusion for those
    if(length(varclass)>0){
      X.da.in1[,varclass]=X[,varclass]+matrix(noise.level*rnorm(length(X[,varclass])),nrow(X),length(varclass))
      X.da.in2[,varclass] =X[,varclass]+matrix(noise.level*rnorm(length(X[,varclass])),nrow(X),length(varclass))
    }

    #find non-continous variables
    varclass=which(ids)

    #random shuffling for those
    if(length(varclass)>0){
      shuffle.pack = sample(1:nrow(X),size=shuffle.rate*nrow(X),replace=FALSE)
      new.order = sample(1:length(shuffle.pack),size=length(shuffle.pack),replace=FALSE)
      X.da.in1[shuffle.pack,varclass]=X[new.order,varclass]
      X.da.in2[shuffle.pack,varclass] = X[new.order,varclass]
    }

    #accounting
    colnames(X.da.in1)=paste('fake1_',1:ncol(X),sep='')
    colnames(X.da.in2)=paste('fake2_',1:ncol(X),sep='')
    X = cbind(X,X.da.in1,X.da.in2)

    X.new1=X.new
    X.new2=X.new

    colnames(X.new1)=paste('fake1_',1:ncol(X.new),sep='')
    colnames(X.new2)=paste('fake2_',1:ncol(X.new),sep='')
    X.new=cbind(X.new,X.new1,X.new2)
  }
  ####################################################################
  ####################################################################

  fits_train=matrix(NA,B,nrow(X))
  fits_test=matrix(NA,B,nrow(X.new))

  for(b in 1:B){ #Bagging
    if(fix.seeds){set.seed(2020+b)}

    #sub-sampling
    sample.in = sample(x=1:length(y),size=sampling.rate*length(y))
    sample.x = sample(x=1:ncol(X),size=mtry*ncol(X))
    y.in = y[sample.in]
    X.in = X[sample.in,sample.x]

    #estimating the model
    baselearner=gbm::gbm(y.in~., data = as.data.frame(cbind(y.in,X.in)),distribution='gaussian',n.trees = n.trees,shrinkage = nu,interaction.depth = tree.depth,bag.fraction = bf)
    #getting in-sample fitted values
    fits_train[b,]=predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X)),X)),n.trees=n.trees)
    #getting out-of-sample predicted values
    fits_test[b,]=predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X.new)),X.new[,sample.x])),n.trees=n.trees)
  }

  #average over the bag
  fit_train=apply(fits_train,2,mean)
  fit_test=apply(fits_test,2,mean)

  # [...] and in the darkness bind them.
  return(c(fit_train,fit_test))
}


###############################################################################
###############################################################################
###############################################################################
####################### MARSquake #############################################
###############################################################################
###############################################################################
###############################################################################



MARSquake = function(y,X,X.new,B=100,mtry=0.8,sampling.rate=.75,data.aug=FALSE,noise.level=0.3,shuffle.rate=0.2,make.sure.it.overfits=TRUE,fix.seeds=TRUE,
                     degree=3,mars.mtry.frac=.3,prune='none',nk=NULL){
  #Row 1 are options related to MARSquake "ensembling"
  #Row 2 are earth's options
  #Keep default values unless yo know what you're doing
  X=as.data.frame(X)
  X.new=as.data.frame(X.new)

  #This modifies Earth so that it randomly allows a fraction (alpha) predictors to be considreed in the forward pass
  allowed <- function(degree, pred, parents, namesx, first)
  {
    # TODO
    allow <- runif(1) < mars.mtry.frac #PROB_OF_ALLOWING_PRED_DURING_EACH_STEP_OF_FORWARD_PASS
    allow # return true if predictor is allowed in this step of the forward pass
  }

  #This block finds out which variables are continous
  ids=rep(NA,ncol(X))
  for(kk in 1:ncol(X)){ids[kk]=length(unique(X[,kk]))==2}
  varclass=which(!ids)

  #This scales the continous variables.
  #This is necessary because we may be adding noise later, which variance is heuristically "tuned" according to X_k ~N(0,1).
  Xall = scale(rbind(X[,varclass],X.new[,varclass]))
  X.new[,varclass] = Xall[(nrow(X)+1):nrow(Xall),]
  X[,varclass] = Xall[1:nrow(X),]

  ##################################################################
  ##################################################################
  # This block implements data augmentation as described in Appendix A.2
  if(data.aug){
    if(fix.seeds){set.seed(1)}

    #creates too copies of X
    X.da.in1 = X
    X.da.in2 = X

    #find continous variables
    varclass=which(!ids)

    #gaussian noise infusion for those
    if(length(varclass)>0){
      X.da.in1[,varclass]=X[,varclass]+matrix(noise.level*rnorm(length(X[,varclass])),nrow(X),length(varclass))
      X.da.in2[,varclass] =X[,varclass]+matrix(noise.level*rnorm(length(X[,varclass])),nrow(X),length(varclass))
    }

    #find non-continous variables
    varclass=which(ids)

    #random shuffling for those
    if(length(varclass)>0){
      shuffle.pack = sample(1:nrow(X),size=shuffle.rate*nrow(X),replace=FALSE)
      new.order = sample(1:length(shuffle.pack),size=length(shuffle.pack),replace=FALSE)
      X.da.in1[shuffle.pack,varclass]=X[new.order,varclass]
      X.da.in2[shuffle.pack,varclass] = X[new.order,varclass]
    }

    #accounting
    colnames(X.da.in1)=paste('fake1_',1:ncol(X),sep='')
    colnames(X.da.in2)=paste('fake2_',1:ncol(X),sep='')
    X = cbind(X,X.da.in1,X.da.in2)

    X.new1=X.new
    X.new2=X.new

    colnames(X.new1)=paste('fake1_',1:ncol(X.new),sep='')
    colnames(X.new2)=paste('fake2_',1:ncol(X.new),sep='')
    X.new=cbind(X.new,X.new1,X.new2)
  }
  ####################################################################
  ####################################################################

  fits_train=matrix(0,B,nrow(X))
  fits_test=matrix(0,B,nrow(X.new))

  for(b in 1:B){ #Bagging
    if(fix.seeds){set.seed(2020+b)}

    #sub-sampling
    sample.in = sample(x=1:length(y),size=sampling.rate*length(y))
    sample.x = sample(x=1:ncol(X),size=mtry*ncol(X))
    y.in = y[sample.in]
    X.in = X[sample.in,sample.x]

    #some rules of thumb to avoid earth causing errors
    if(is.null(nk)){nk=round(min(900,length(y.in)*0.75))}
    else{nk=round(min(c(900,length(y.in)*0.75,nk)))}

    if(!make.sure.it.overfits){
      #estimating the model
      baselearner=earth::earth(y.in~., data = as.data.frame(cbind(y.in,X.in)),degree=degree,pmethod=prune,nk=nk,thresh=0,trace=0,allowed=allowed,penalty=-1)
      #getting in-sample fitted values
      fits_train[b,]=predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X)),X)))
      #getting out-of-sample predicted values
      fits_test[b,]=predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X.new)),X.new[,sample.x])))
    }

    #iniate
    y.in.ori=y.in
    if(make.sure.it.overfits){
      #initiate
      step=0
      prev.rsq=0
      for(step in 1:6){ #not more than 5 additional attemps at making earth overfit (see appendix A.2)
        y.in=y.in-fits_train[b,sample.in]

        baselearner=earth::earth(y.in~., data = as.data.frame(cbind(y.in,X.in)),degree=degree,pmethod=prune,nk=nk,thresh=0,trace=0,allowed=allowed,penalty=-1)
        fits_train[b,]=fits_train[b,]+predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X)),X)))
        fits_test[b,]=fits_test[b,]+predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X.new)),X.new[,sample.x])))

        if(r.squared(fits_train[b,sample.in], y.in.ori)>0.9){break}
        if(r.squared(fits_train[b,sample.in], y.in.ori)<prev.rsq){
          #undo then break
          fits_train[b,]=fits_train[b,]-predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X)),X)))
          fits_test[b,]=fits_test[b,]-predict(baselearner,newdata=as.data.frame(cbind(rep(NA,nrow(X.new)),X.new[,sample.x])))
          break
        }
        prev.rsq=r.squared(fits_train[b,sample.in], y.in.ori)
      }
    }
  }

#average over the bag
fit_train=apply(fits_train,2,mean)
fit_test=apply(fits_test,2,mean)

# [...] and in the darkness bind them.
return(c(fit_train,fit_test))
}

