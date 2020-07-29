# Librairie Requise
install.packages("recommenderlab")
library(recommenderlab)
set.seed(12345)


##########################
# ?valuations Explicites #
##########################

# Import des donn?es
data("MovieLense")

# Exploration des donn?es
head <- MovieLense[c(1:5), c(1:3)]  
as(head, "matrix")
image(MovieLense[1:100, 1:100]) 

# Histogramme des evaluations
hist(getRatings(MovieLense), 
     breaks=15,
     xlab='?valuations',
     main='Histogramme des ?valuations')

# S?paration du jeu de donn?es en train et test gr?ce ? la fonction evaluationScheme
e_scheme <- evaluationScheme(MovieLense,method="split", train=0.7, given=4, goodRating=4)       

# Modeles disponibles pour "realRatingMatrix"
names(recommenderRegistry$get_entries(dataType = "realRatingMatrix"))

##############################################
# Premier mod?le - films populaires

start.time <- Sys.time() 
model_pop <- Recommender(MovieLense[1:500], method = "POPULAR")              
                                                                  
recom_pop <- predict(model_pop, MovieLense[501:502], n=5)          
as(recom_pop, "list")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

####################################################
# Second mod?le - User-based collaborative filtering


start.time <- Sys.time()                                                            
UBCF_params <- list(method="cosine", nn = 10, sample = FALSE, normalize="center")

model_UBCF <- Recommender(getData(e_scheme, "train"), method="UBCF", parameter= UBCF_params)                 
pred_UBCF <- predict(model_UBCF, getData(e_scheme, "known"), type="ratings")                              
error_UBCF <- calcPredictionAccuracy(pred_UBCF, getData(e_scheme, "unknown"))
error_UBCF
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

#######################################################
# Troisi?me mod?le - Item-based collaborative filtering
       
start.time <- Sys.time()                                                              
IBCF_params <- list(method="cosine", k = 10, normalize="center")

model_IBCF <- Recommender(getData(e_scheme, "train") , method="IBCF", parameter= IBCF_params)
pred_IBCF <- predict(model_IBCF, getData(e_scheme, "known"), type="ratings")
error_IBCF <- calcPredictionAccuracy(pred_IBCF, getData(e_scheme, "unknown"))
error_IBCF
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
#######################################################
# Quatri?me mod?le - SVD (column mean imputation)
start.time <- Sys.time()  
svd_params <- list(k=13, maxiter=200, normalize="center")

model_svd <- Recommender(getData(e_scheme, "train"), method="SVD", parameter=svd_params)

pred_svd <- predict(model_svd, getData(e_scheme, "known"), type="ratings")
error_svd <- calcPredictionAccuracy(pred_svd, getData(e_scheme, "unknown"))
error_svd
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
#######################################################
# Cinqui?me mod?le - FunkSVD 
start.time <- Sys.time() 
funk_params <- list(k = 13, lambda = 0.001, min_epochs=50, max_epochs=100, normalize="center")

model_funk <- Recommender(getData(e_scheme, "train"), method="SVDF", parameter = funk_params)
pred_funk <- predict(model_funk, getData(e_scheme, "known"), type="ratings")
error_funk <- calcPredictionAccuracy(pred_funk, getData(e_scheme, "unknown"))
error_funk
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
#######################################################
# Septi?me mod?le - Syst?me de recommandation hybride 
start.time <- Sys.time() 
hybrid_model <- HybridRecommender(Recommender(MovieLense, method="POPULAR"),
                                  Recommender(MovieLense, method="UBCF"),
                                  Recommender(MovieLense, method="IBCF"),
                                  Recommender(MovieLense, method="RERECOMMEND"),
                                  Recommender(MovieLense, method="RANDOM"),
                                  weights = c(0.3, 0.2, 0.3, 0.1, 0.1)
                                 )

getList(predict(hybrid_model, 501:502, MovieLense, n=5))
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

######################################################
# ?valuations des recommandations
start.time <- Sys.time() 
c_scheme <- evaluationScheme(MovieLense, method="cross", k=2, given=4, goodRating=4)
compared_algos <- list( "random items" = list(name="RANDOM", param=NULL),
                        "popular items" = list(name="POPULAR", param=list(normalize="center")),
                        "user-based CF" = list(name="UBCF", param=list(nn=40, method="cosine",normalize="center")),
                        "item-based CF" = list(name="IBCF", param=list(k=40,normalize="center")),
                        "SVD" = list(name="SVD", param=list(k=13,normalize="center")),
                        "FUNKSVD" = list(name="SVDF", param=list(k=13,normalize="center"))
                        )

# Note: Meme si le k-fold est petit, cette commande prend du temps ? executer
# Pour top-N recommandations
topn_comp <- evaluate(c_scheme, compared_algos, type="topNList", n=c(1,3,5,10,15,20))
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
# Graphique de la courbe ROC
plot(topn_comp, annotate=c(1,6,2), legend='topleft')

# Graphique de precision/rappel
plot(topn_comp, "prec/rec", annotate=c(1,2,5))

# Pour evaluations
ratings_comp <- evaluate(c_scheme, compared_algos, type="ratings")

# Histogramme 
# Librairies pour couleurs
install.packages("RColorBrewer")
library(RColorBrewer)
coul <- brewer.pal(6, "Set3")
plot(ratings_comp, col=coul)


##################################
# M?thode ALS avec rrecsys

install.packages("rrecsys")
library(rrecsys)

start.time <- Sys.time()
# Transformation de realRatingMatrix
mat_ml <- as(MovieLense[1:150], "matrix")

# Definition des donnees (transformation)
ml_rrec <- defineData(mat_ml, sparseMatrix=FALSE, binary=FALSE, minimum=1, maximum=5)

# Definition de l'evaluation du modele
als_cv <- evalModel(ml_rrec, folds = 3)

# Evaluation des predictions
eval_als <- evalRec(als_cv, "wALS", k = 15, delta = 0.04, scheme="uni", positiveThreshold=4, topN=5)

eval_als
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
##########################
# ?valuations implicites #
##########################
 
# Import des donn?es
rawdata <- read.csv("C:/Users/admin/Desktop/HEC H20/Stats avance/Travail de session/markdown-rapport/lastfm.csv")

# Exploration des donn?es
head(rawdata)

# Nombre unique de client 
length(unique(rawdata[['userid']]))
# Nombre d'artistes
length(unique(rawdata[['artistid']]))

# Pour notre analyse, on retire le nom de l'artiste
lastfm <- rawdata[,-3]

# Construction de la matrice d'interaction. 
matlastfm <- sparseMatrix(i = as.integer(lastfm$userid),
                          j = as.integer(lastfm$artistid),
                          x = lastfm$plays)

# On retire les usagers qui ont moins de 15 observations d'ecoute 
matlastfm <- matlastfm[tabulate(summary(matlastfm)$i) > 15, , drop=FALSE]
dim(matlastfm)

# Base sur Hu, et al., nous ajoutons une mesure de confiance

adjust_confidence <- function(x, alpha){
  mat_conf <- x
  mat_conf@x <- 1 + alpha * x@x
  mat_conf
}

conflastfm <- adjust_confidence(matlastfm, 0.1)

# Transformation de la matrice en "realRatingMatrix"
last_fm <- as(conflastfm, "realRatingMatrix")

set.seed(12345)
i_scheme = evaluationScheme(last_fm[1:250], method="split", train=0.7, given=8)

# Premier mod?le - ALS

params_als <- list(lambda = 0.1,
                   n_factors = 50,
                   n_iterations = 30,
                   seed=12345)


model_ALS <- Recommender(getData(i_scheme, "train"), method="ALS", parameter=params_als)
pred_als <- predict(model_ALS, getData(i_scheme, "known"), type="ratings")


error_als <- calcPredictionAccuracy(pred_als, getData(i_scheme, "unknown"))
error_als

library(rsparse)

train <- matlastfm[1:5000,]
test <- matlastfm[5001:nrow(matlastfm),]
start.time <- Sys.time()
model_alsr <- WRMF$new(rank=15, feedback="implicit", lambda=0.1, solver="cholesky")
fit_als <- model_alsr$fit_transform(train, n_iter=50)
pred_alsr <- model_alsr$predict(test, k=20, not_recommend=NULL)
mean(ap_k(pred_alsr, actual = test))

model_alsr$components
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
############################
# Second modele - Bayesian Personalized Ranking

data(ml100k)

ml_bpr <- ml100k[1:25,1:75]

# Definition des donnees (transformation)
ml_rrec <- defineData(ml_bpr, binary=TRUE)

# Definition de l'evaluation du modele
eval_als <- evalModel(ml_rrec, folds = 2)

bpr <- rrecsys(ml_rrec, "BPR", k = 10, randomInit = FALSE, regU = .0025, regI = .0025, regJ = 0.0025, updateJ = TRUE)

# Evaluation des predictions
pred_als <- evalRec(eval_als, "BPR", k = 5, topN=3)
model_bpr <- recc
