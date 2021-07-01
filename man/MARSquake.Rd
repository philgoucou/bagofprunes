\name{MARSquake}
\alias{MARSquake}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{MARSquake
%%  ~~function to do ... ~~
}
\description{
MARSquake is bagged and perturbed ensemble of completely overfitting MARS base learners. There is a data augmentation option which can be activated to enhanced perturbation's potential when only few features are available. As argued in the paper referenced below, proper randomization will implicitly deliver the optimally stopped MARS. Hence, like for Random Forests, cross-validation is not necessary to avoid overfitting. However, it is not impossible that tweaking some randomization hyperparameters could marginally increase performance. Currently, the function only supports regression. For more details and thorough explanations see appendix A.2 in the paper.
}
\usage{
MARSquake(y,X,X.new,B=100,mtry=0.8,sampling.rate=.75,
        data.aug=FALSE,noise.level=0.3,shuffle.rate=0.2,
        make.sure.it.overfits=TRUE,fix.seeds=TRUE,
        degree=3,mars.mtry.frac=.3,prune='none',nk=NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{y}{training target}
  \item{X}{training features}
  \item{X.new}{features for test set prediction}
  \item{B}{number of ensemble members}
  \item{mtry}{fraction of randomly selected features considered for each base learners}
  \item{sampling.rate}{subsampling rate}
  \item{data.aug}{Should we augment the feature matrix with two noisy pseudo-carbo copies of X?}
  \item{noise.level}{Standard deviation of the Gaussian noise added to the continuous variables copies of X (when data.aug=TRUE). Note that X's are standardized beforehand.}
  \item{shuffle.rate}{Controls the fraction of observations being shuffled for non-continuous regressors when data.aug=TRUE.}
  \item{make.sure.it.overfits}{If TRUE, this option partially forces "earth" to overfit (in-sample) if it is recalcitrant to do so (happens when features are scarce).}
\item{mars.mtry.frac}{Controls the fraction of randomly selected features as potential candidates at each step in "earth" forward pass. Analogous to "mtry" in Random Forests.}
\item{degree}{Option on base learners. See "earth"package.}
\item{prune}{Option on base learners. See "earth"package. The whole point of the related paper is that there is little to no benefits in using anything other than "none". However, in data sets where perfect randomization seems unattainable, this could be worth exploring.}
\item{nk}{Option on base learners. See "earth" package. Keep NULL unless you really know what you are doing.}
\item{fix.seeds}{for replicability}

}

\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
The function returns a vector binding (in this order) training set fitted values and test set predictions.
}
\references{
Related paper is available at https://arxiv.org/abs/2008.07063.
}
\author{
Philippe Goulet Coulombe
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
set.seed(200905)
K=5
dat=matrix(rnorm(K*200),200,K)
test=101:200
train=1:100

X=dat[train,2:K]
X.new=dat[test,2:K]
y=crossprod(t(X),rep(1,(K-1)))+dat[train,1]
y.new=crossprod(t(X.new),rep(1,(K-1)))+dat[test,1]

output=MARSquake(y,X,X.new)
benchmark = sqrt(mean((mean(y)-y.new)^2))
sqrt(mean((output[test]-y.new)^2))/benchmark

output=MARSquake(y,X,X.new,data.aug =TRUE)
sqrt(mean((output[test]-y.new)^2))/benchmark

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
