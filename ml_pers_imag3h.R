### Script comparación resultados de diversos métodos de aprendizaje automático en detección de ríos atmosféricos.
### El imput es el de las imágenes de persistencia 0 y 1 dimensionales



####1 Carga de librerías.

library(readr)
library(data.table)
library(ROSE)
library(class) #Se necesita para knn
library(pROC)
library(e1071)
library(MASS)


###MIRAR SI SE HA PUESTO EL ROWNAME COMO COLUMNA EN LOS READ_csv, EN FUNCION DE ESO ELIMINAR LA PRIMERA COLUMNA O NO EN EL MERGE (PARA AMBOS DATA TABLE)
###TAMBIEN SI SE PUSO EL ROWNAME COMO COLUMNA CAMBIAR LO TRATAR POR SEPAREADO A
#LOS NNOMBRES DE COLUMNA 1,2 POR LOS NOMBRES DE COLUMNA 1,2,3




##### 2. Carga y tratamiento de datos



persim<-as.data.table(read_csv("mestrado/tfm-solicitude bolsa/piton/persim_estable45.csv", 
                            col_types = cols(...1 = col_skip(), ano = col_double(), 
                                            t = col_double())))

etiquetas<-as.data.table(read_csv("mestrado/tfm-solicitude bolsa/piton/etiquetas_merra.csv", 
                              col_types = cols(...1 = col_skip(), ano = col_double(), 
                                        t = col_double())))

summary(etiquetas)

etiquetas[,AR:=as.factor(etiquetas[,AR])] # Variable AR la convertimos en factor

#cambiamos nombres variables año e imagen 
colnames(persim)[1:2]<-colnames(etiquetas)[1:2]

#cambiamos combres variables de imágenes de persistencia
colnames(persim)[-c(1,2)]<-paste0("X",colnames(persim)[-c(1,2)])


#persim<-persim[, 1:137] #solo dim 1
#persim<-persim[, -(3:127)] #solo dim 0


#Unimos las etiquetas AR a las imágenes de persistencia equilibradas
dataset_desequilibrado<-merge(x=etiquetas[,], y=persim[,], by=colnames(etiquetas)[1:2], all.y=FALSE, all.x=FALSE)

table(dataset_desequilibrado[, AR])

#formula de los métodos q vamos a usar
formula<-paste( "AR", paste0(colnames(dataset_desequilibrado)[-c(1,2,3)] , collapse = ' + '), sep = ' ~ ')

#en vez de colnames(dataset_desequilibrado)[-c(1,2,3)] se podría haber puesto algo de este estilo filtrando por valor de colname y no por índice my_vector[! my_vector %in% c('a', 'b, 'c')]


# Resampleamos eliminando datos que tienen AR=0 hasta obtener clase AR=1 el 48% de los datos
 dataset_equilibrado <-as.data.table(ovun.sample(AR~., data=dataset_desequilibrado,
                                      p=0.48, 
                                     seed=1, method="under")$data)

rm(persim) #eliminamos datatables q ya no son necesarios
gc()
class(dataset_equilibrado)
 
table(dataset_equilibrado[,AR])
prop.table(table(dataset_equilibrado[,AR]))
prop.table(table(dataset_desequilibrado[,AR]))


##### 3 Dividimos aleatoriamente las filas en 5 folds de tamaño casi igual
nfilas_equilibrado=dim(dataset_equilibrado)[1]
indices<-1:nfilas_equilibrado

set.seed(2)

permutacion_indices<-sample(indices, size=nfilas_equilibrado)

fold<-vector('list',length = 5)
fold[[1]]<-permutacion_indices[1:floor(0.20*nfilas_equilibrado)]
fold[[2]]<-permutacion_indices[((floor(0.20*nfilas_equilibrado))+1):(2*floor(0.20*nfilas_equilibrado))]
fold[[3]]<-permutacion_indices[((2*floor(0.20*nfilas_equilibrado))+1):(3*floor(0.20*nfilas_equilibrado))]
fold[[4]]<-permutacion_indices[((3*floor(0.20*nfilas_equilibrado))+1):(4*floor(0.20*nfilas_equilibrado))]
fold[[5]]<-permutacion_indices[((4*floor(0.20*nfilas_equilibrado))+1):nfilas_equilibrado]
fold





####  4 KNN 
### para cada i vamos rotando los conjuntos de validación (1 fold), test (1 fold), entreno (3 fold)
## Entrenamos el modelo en el conjunto entreno y calculamos para diferentes valores de los hiperparámetros 
### la accuracy en el conjunto de validación. Nos quedamos con los valores de hiperparámetros que maximizan esa
##accuracy en el conjunto de entrenamiento. Luego calculamos accuracy, precision y sensitividad en conjunto test.
## # Finalmente imprimimos estos vectores de acc, prec y sens y sus medias.
### Hiperparámetros: k \in \{1,2,..., 10, 15, 20, ..., 75\}
### En la definicion de datos_t/v/test_estand vemos si escalamos (o no) juntos (o no) los datos de
### entrenamiento/validacion/test

accuracy<-vector(mode = "double", length=5)
precision<-vector(mode = "double", length=5)
sensitividad<-vector(mode = "double", length=5)

x_escalado<-scale(dataset_equilibrado[,-c(1,2,3)])

k_fold<-vector(mode = "integer", length=5)
for (i in 1:5){

  conjunto_validacion<-dataset_equilibrado[fold[[i]],]
  x_val<-x_escalado[fold[[i]],]
  if (i!=5){
    conjunto_test<-dataset_equilibrado[fold[[i+1]],]
    x_test<-x_escalado[fold[[i+1]],]
    if (i==4){
      conjunto_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]]),]
      x_entreno<-x_escalado[c(fold[[1]], fold[[2]], fold[[3]]),]
      cat(i, i+1, 1,2,3, "\n")
    }
    if (i==3){
      conjunto_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]]),]
      x_entreno<-x_escalado[c(fold[[5]], fold[[1]], fold[[2]]),]
      cat(i, i+1, 5,1,2, "\n")
    }
    if (i==2){
      conjunto_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]]),]
      x_entreno<-x_escalado[c(fold[[4]], fold[[5]], fold[[1]]),]
      cat(i, i+1, 4,5,1, "\n")
    }
    if (i==1){
      conjunto_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]]),]
      x_entreno<-x_escalado[c(fold[[3]], fold[[4]], fold[[5]]),]
      cat(i, i+1, 3,4,5, "\n")
    }    
  }
  if (i==5){
    conjunto_test<-dataset_equilibrado[fold[[1]],]
    x_test<-x_escalado[fold[[1]],]
    conjunto_entreno<-dataset_equilibrado[c(fold[[2]], fold[[3]], fold[[4]]),]
    x_entreno<-x_escalado[c(fold[[2]], fold[[3]], fold[[4]]),]
    cat(i, 1, "\n")
  }


 accuracy_validacion_1<-vector(mode = "double", length=75) ##vector con la acuracy en 
 
 ######escalamos (o no) los datos por separado (o no)
 datos_t_estand<-conjunto_entreno[,-c(1,2,3)] #x_entreno# scale(conjunto_entreno[,-c(1,2,3)]) 
 datos_v_estand<-conjunto_validacion[,-c(1,2,3)]#x_val ## scale(conjunto_validacion[,-c(1,2,3)])
 datos_test_estand<- conjunto_test[,-c(1,2,3)] #x_test##scale(conjunto_test[,-c(1,2,3)])
 
 y_validacion<-conjunto_validacion[, AR]==1
 y_entreno<-conjunto_entreno[, AR]==1
 y_test<-conjunto_test[, AR]==1
 
 set.seed(456)
  for (k in c(1:10, seq(15, 75, 5))){

      modeloi<- knn(train=datos_t_estand, test=datos_v_estand,cl=y_entreno , k = k,use.all=FALSE) 
      accuracy_validacion_1[k]<- mean(modeloi == y_validacion)
  }
  
  cat(accuracy_validacion_1, " \n")
 
  k=which.max(accuracy_validacion_1)
  cat(k, accuracy_validacion_1[k], " \n")

  k_fold[i]<-k

  
  pred_test<-knn(train=datos_t_estand, test=datos_test_estand,cl=y_entreno , k = k,use.all=FALSE, prob=FALSE) 

  matriz_confusion<-table(y_test, pred_test) 
  
  print(matriz_confusion)
  
  accuracy[i]=(matriz_confusion[1,1]+matriz_confusion[2,2])/sum(matriz_confusion)
  sensitividad[i]=matriz_confusion[2,2]/sum(matriz_confusion[2,])
  precision[i]=matriz_confusion[2,2]/sum(matriz_confusion[,2])
}    
 cat(k_fold[[1]],k_fold[[2]],k_fold[[3]],k_fold[[4]],k_fold[[5]], "\n")
cat(accuracy, mean(accuracy), "\n")
cat(sensitividad, mean(sensitividad), "\n")
cat(precision, mean(precision), "\n")



















### 5.1 SVM radial
### para cada i vamos rotando los conjuntos de validación (1 fold), test (1 fold), entreno (3 fold)
## Entrenamos el modelo en el conjunto entreno y calculamos para diferentes valores de los hiperparámetros 
### la accuracy en el conjunto de validación. Nos quedamos con los valores de hiperparámetros que maximizan esa
##accuracy en el conjunto de entrenamiento. Luego calculamos accuracy, precision y sensitividad en conjunto test.
## # Finalmente imprimimos estos vectores de acc, prec y sens y sus medias.
### Hiperparámetros: C y gamma
### En la definicion de scalar vemos si el objeto SVM escala (o no) los datos
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM

accuracy_radial<-vector(mode = "double", length=5)
precision_radial<-vector(mode = "double", length=5)
sensitividad_radial<-vector(mode = "double", length=5)
C_fold_radial<-vector(mode = "double", length=5)
gamma_fold<-vector(mode = "double", length=5)
scalar<-FALSE#TRUE
x_escalado<-dataset_equilibrado[,-c(1,2,3)]
  #scale(dataset_equilibrado[,-c(1,2,3)]) #escalamos juntos todos los datos
  
  
  #

#
  #



for (i in 1:5){
  
  
  y_val<-dataset_equilibrado[fold[[i]],AR]
  x_val<-x_escalado[fold[[i]],]
  if (i!=5){
    
    y_test<-dataset_equilibrado[fold[[i+1]],AR]
    x_test<-x_escalado[fold[[i+1]],]
    if (i==4){
      
      y_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]]),AR]
     
      x_entreno<-x_escalado[c(fold[[1]], fold[[2]], fold[[3]]),]
      cat(i, i+1, 1,2,3, "\n")
    }
    if (i==3){
      
      x_entreno<-x_escalado[c(fold[[5]], fold[[1]], fold[[2]]),]
      y_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]]),AR]

      
      cat(i, i+1, 5,1,2, "\n")
    }
    if (i==2){
      
      y_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]]),AR]
      x_entreno<-x_escalado[c(fold[[4]], fold[[5]], fold[[1]]),]
      cat(i, i+1, 4,5,1, "\n")
    }
    if (i==1){
      
      y_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]]),AR]
      x_entreno<-x_escalado[c(fold[[3]], fold[[4]], fold[[5]]),]
      cat(i, i+1, 3,4,5, "\n")
    }    
  }
  if (i==5){

    y_test<-dataset_equilibrado[fold[[1]],AR]
    x_test<-x_escalado[fold[[1]],]
    y_entreno<-dataset_equilibrado[c(fold[[2]], fold[[3]], fold[[4]]),AR]
    x_entreno<-x_escalado[c(fold[[2]], fold[[3]], fold[[4]]),]
    cat(i, 1, "\n")
  }

  acu<-matrix(NA, nrow=0, ncol=3)
  colnames(acu)<-c("gama", "c", "acc")
  for (gama in seq(0.0055, 0.013, 0.0015)) {
    for (c in c(1,2,4,8)){
  
  qdebien<-svm(x=x_entreno, y=y_entreno, scale=scalar, type="C-classification", kernel="radial", cost=c, gamma=gama)
  predictingcontent<- predict(qdebien, x_val) #predict(qdebien, scale(conjunto_validacion[,-c(1,2,3)]))
  acu<-rbind(acu, c(gama, c, mean(predictingcontent == y_val)))
    }
  }
  gamba<-acu[which.max(acu[,3]),1]
  cos<-acu[which.max(acu[,3]),2]  
  
  C_fold_radial[i]<-cos
  gamma_fold[i]<-gamba
    
  svm_definitivo<-svm(x=x_entreno, y=y_entreno, scale=scalar, type="C-classification", kernel="radial", cost=cos, gamma=gamba)
  
  
  pred_test<-predict(svm_definitivo, x_test)#predict(svm_definitivo, ¿scale?conjunto_test[,-c(1,2,3)])
  
  
  
  confu_test<-table(y_test, pred_test)
  accuracy_radial[i]=(confu_test[1,1]+confu_test[2,2])/sum(confu_test)
  sensitividad_radial[i]=confu_test[2,2]/sum(confu_test[2,])
  precision_radial[i]=confu_test[2,2]/sum(confu_test[,2])
  
}

 cat(accuracy_radial, mean(accuracy_radial), "\n")
cat(sensitividad_radial, mean(sensitividad_radial), "\n")
cat(precision_radial, mean(precision_radial), "\n")
cat(gamma_fold, "\n")
cat(C_fold_radial, "\n")

























### 5.2 SVM lineal
### para cada i vamos rotando los conjuntos de validación (1 fold), test (1 fold), entreno (3 fold)
## Entrenamos el modelo en el conjunto entreno y calculamos para diferentes valores de los hiperparámetros 
### la accuracy en el conjunto de validación. Nos quedamos con los valores de hiperparámetros que maximizan esa
##accuracy en el conjunto de entrenamiento. Luego calculamos accuracy, precision y sensitividad en conjunto test.
## # Finalmente imprimimos estos vectores de acc, prec y sens y sus medias.
### Hiperparámetros: C 
### En la definicion de scalar vemos si el objeto SVM escala (o no) los datos
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM


accuracy_lineal<-vector(mode = "double", length=5)
precision_lineal<-vector(mode = "double", length=5)
sensitividad_lineal<-vector(mode = "double", length=5)
C_fold_lineal<-vector(mode = "double", length=5)
scalar<-FALSE#TRUE
x_escalado<-scale(dataset_equilibrado[,-c(1,2,3)])

for (i in 1:5){
  
  y_val<-dataset_equilibrado[fold[[i]],AR]
  x_val<-x_escalado[fold[[i]],]
  if (i!=5){
    y_test<-dataset_equilibrado[fold[[i+1]],AR]
    x_test<-x_escalado[fold[[i+1]],]
    if (i==4){
      y_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]]),AR]
      x_entreno<-x_escalado[c(fold[[1]], fold[[2]], fold[[3]]),]
      cat(i, i+1, 1,2,3, "\n")
    }
    if (i==3){
      x_entreno<-x_escalado[c(fold[[5]], fold[[1]], fold[[2]]),]
      y_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]]),AR]
      cat(i, i+1, 5,1,2, "\n")
    }
    if (i==2){
      y_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]]),AR]
      x_entreno<-x_escalado[c(fold[[4]], fold[[5]], fold[[1]]),]
      cat(i, i+1, 4,5,1, "\n")
    }
    if (i==1){
      y_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]]),AR]
      x_entreno<-x_escalado[c(fold[[3]], fold[[4]], fold[[5]]),]
      cat(i, i+1, 3,4,5, "\n")
    }    
    }
  if (i==5){
      y_test<-dataset_equilibrado[fold[[1]],AR]
      x_test<-x_escalado[fold[[1]],]
      y_entreno<-dataset_equilibrado[c(fold[[2]], fold[[3]], fold[[4]]),AR]
      x_entreno<-x_escalado[c(fold[[2]], fold[[3]], fold[[4]]),]
      cat(i, 1, "\n")
  }
  

  
  set.seed(45)
  

  acu<-matrix(NA, nrow=0, ncol=2)
  colnames(acu)<-c("c", "acc")
    for (c in c(1:10,14,18,23,28,33,39,47,55,64,73,82)){
      
      qdebien<-svm(x=x_entreno, y=y_entreno, scale=scalar,kernel="linear", type="C-classification",cost=c)
      predictingcontent<-predict(qdebien, x_val)
      acu<-rbind(acu, c(c, mean(predictingcontent == y_val)))
    
  }
  cos<-acu[which.max(acu[,2]),1]  
  
  C_fold_lineal[i]<-cos

  svm_definitivo<-svm(x=x_entreno, y=y_entreno, scale=scalar,kernel="linear", type="C-classification", cost=cos)
  
  
  
  pred_test<-predict(svm_definitivo, x_test) #1*(prob_test>=rep(umbral, length(prob_test)))
  
  
  confu_test<-table(y_test, pred_test)
  accuracy_lineal[i]=(confu_test[1,1]+confu_test[2,2])/sum(confu_test)
  sensitividad_lineal[i]=confu_test[2,2]/sum(confu_test[2,])
  precision_lineal[i]=confu_test[2,2]/sum(confu_test[,2])
  
}

cat(accuracy_lineal, mean(accuracy_lineal), "\n")
cat(sensitividad_lineal, mean(sensitividad_lineal), "\n")
cat(precision_lineal, mean(precision_lineal), "\n")
cat(C_fold_lineal, "\n")

























### 5.3 SVM polinomial 
### para cada i vamos rotando los conjuntos de validación (1 fold), test (1 fold), entreno (3 fold)
## Entrenamos el modelo en el conjunto entreno y calculamos para diferentes valores de los hiperparámetros 
### la accuracy en el conjunto de validación. Nos quedamos con los valores de hiperparámetros que maximizan esa
##accuracy en el conjunto de entrenamiento. Luego calculamos accuracy, precision y sensitividad en conjunto test.
### Finalmente imprimimos estos vectores de acc, prec y sens y sus medias.
### Hiperparámetros: C, gamma, grado, intercepto
### En la definicion de scalar vemos si el objeto SVM escala (o no) los datos
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM

accuracy_polinomial<-vector(mode = "double", length=5)
precision_polinomial<-vector(mode = "double", length=5)
sensitividad_polinomial<-vector(mode = "double", length=5)
C_fold_polinomial<-vector(mode = "double", length=5)
gamma_fold_polinomial<-vector(mode = "double", length=5)
intercept_fold_polinomial<-vector(mode = "double", length=5)
grado_fold_polinomial<-vector(mode = "double", length=5)
scalar<-FALSE#TRUE
x_escalado<-scale(dataset_equilibrado[,-c(1,2,3)])


for (i in 1:5){
  
  y_val<-dataset_equilibrado[fold[[i]],AR]
  x_val<-x_escalado[fold[[i]],]
  if (i!=5){
    y_test<-dataset_equilibrado[fold[[i+1]],AR]
    x_test<-x_escalado[fold[[i+1]],]
    if (i==4){
      y_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]]),AR]
      x_entreno<-x_escalado[c(fold[[1]], fold[[2]], fold[[3]]),]
      cat(i, i+1, 1,2,3, "\n")
    }
    if (i==3){
      x_entreno<-x_escalado[c(fold[[5]], fold[[1]], fold[[2]]),]
      y_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]]),AR]
      cat(i, i+1, 5,1,2, "\n")
    }
    if (i==2){
      y_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]]),AR]
      x_entreno<-x_escalado[c(fold[[4]], fold[[5]], fold[[1]]),]
      cat(i, i+1, 4,5,1, "\n")
    }
    if (i==1){
      y_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]]),AR]
      x_entreno<-x_escalado[c(fold[[3]], fold[[4]], fold[[5]]),]
      cat(i, i+1, 3,4,5, "\n")
    }    
  }
  if (i==5){
    y_test<-dataset_equilibrado[fold[[1]],AR]
    x_test<-x_escalado[fold[[1]],]
    y_entreno<-dataset_equilibrado[c(fold[[2]], fold[[3]], fold[[4]]),AR]
    x_entreno<-x_escalado[c(fold[[2]], fold[[3]], fold[[4]]),]
    cat(i, 1, "\n")
  }
  
  set.seed(44)
  #predict(supor, x_test)
  acu<-matrix(NA, nrow=0, ncol=5)
  colnames(acu)<-c("c","gama", "intercept", "grado", "acc")
  for (c in c(1,2,4,8)){
    for (gama in seq(0.0055, 0.013,0.0015)){
      for (intercept in seq(0, 0.6, 0.2)){
        for (grado in 2:3){
    
    qdebien<-svm(x=x_entreno, y=y_entreno, scale=scalar, kernel="polynomial",type="C-classification",cost=c, gamma=gama, coef0=intercept,  degree=grado)
    predictingcontent<-predict(qdebien, x_val)
    acu<-rbind(acu, c(c, gama, intercept, grado, mean(predictingcontent == y_val)))
        }
    }
    }
  }
  cos<-acu[which.max(acu[,5]),1]  
  gamba<-acu[which.max(acu[,5]),2]
  intercepto<-acu[which.max(acu[,5]),3]
  grao<-acu[which.max(acu[,5]),4]
  C_fold_polinomial[i]<-cos
  intercept_fold_polinomial[i]<-intercepto
  grado_fold_polinomial[i]<-grao
  gamma_fold_polinomial[i]<-gamba

  
  
  svm_definitivo<-svm(x=x_entreno, y=y_entreno, scale=scalar, kernel="polynomial",type="C-classification", cost=cos, gamma=gamba, coef0=intercepto , degree=grao )
  

  
  

  
  
  pred_test<-predict(svm_definitivo, x_test) #1*(prob_test>=rep(umbral, length(prob_test)))
  
  
  confu_test<-table(y_test, pred_test)
  accuracy_polinomial[i]=(confu_test[1,1]+confu_test[2,2])/sum(confu_test)
  sensitividad_polinomial[i]=confu_test[2,2]/sum(confu_test[2,])
  precision_polinomial[i]=confu_test[2,2]/sum(confu_test[,2])
  
}

cat(accuracy_polinomial, mean(accuracy_polinomial), "\n")
cat(sensitividad_polinomial, mean(sensitividad_polinomial), "\n")
cat(precision_polinomial, mean(precision_polinomial), "\n")
cat(C_fold_polinomial, "\n")
cat(gamma_fold_polinomial, "\n")
cat(intercept_fold_polinomial, "\n")
cat(grado_fold_polinomial, "\n")









### 5.4 SVM sigmoidal
### para cada i vamos rotando los conjuntos de validación (1 fold), test (1 fold), entreno (3 fold)
## Entrenamos el modelo en el conjunto entreno y calculamos para diferentes valores de los hiperparámetros 
### la accuracy en el conjunto de validación. Nos quedamos con los valores de hiperparámetros que maximizan esa
##accuracy en el conjunto de entrenamiento. Luego calculamos accuracy, precision y sensitividad en conjunto test.
## # Finalmente imprimimos estos vectores de acc, prec y sens y sus medias.
### Hiperparámetros: C, gamma, intercepto
### En la definicion de scalar vemos si el objeto SVM escala (o no) los datos
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM


accuracy_sigmoid<-vector(mode = "double", length=5)
precision_sigmoid<-vector(mode = "double", length=5)
sensitividad_sigmoid<-vector(mode = "double", length=5)
C_fold_sigmoid<-vector(mode = "double", length=5)
gamma_fold_sigmoid<-vector(mode = "double", length=5)
intercept_fold_sigmoid<-vector(mode = "double", length=5)
scalar<-FALSE

x_escalado<-scale(dataset_equilibrado[,-c(1,2,3)]) #escalamos juntos todos los datos
#dataset_equilibrado[,-c(1,2,3)] #opcion sin escalar
#


for (i in 1:5){
  
  y_val<-dataset_equilibrado[fold[[i]],AR]
  x_val<-x_escalado[fold[[i]],]
  if (i!=5){
    y_test<-dataset_equilibrado[fold[[i+1]],AR]
    x_test<-x_escalado[fold[[i+1]],]
    if (i==4){
      y_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]]),AR]
      x_entreno<-x_escalado[c(fold[[1]], fold[[2]], fold[[3]]),]
      cat(i, i+1, 1,2,3, "\n")
    }
    if (i==3){
      x_entreno<-x_escalado[c(fold[[5]], fold[[1]], fold[[2]]),]
      y_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]]),AR]
      
      cat(i, i+1, 5,1,2, "\n")
    }
    if (i==2){
      y_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]]),AR]
      x_entreno<-x_escalado[c(fold[[4]], fold[[5]], fold[[1]]),]
      cat(i, i+1, 4,5,1, "\n")
    }
    if (i==1){
      y_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]]),AR]
      x_entreno<-x_escalado[c(fold[[3]], fold[[4]], fold[[5]]),]
      cat(i, i+1, 3,4,5, "\n")
    }    
  }
  if (i==5){
    y_test<-dataset_equilibrado[fold[[1]],AR]
    x_test<-x_escalado[fold[[1]],]
    y_entreno<-dataset_equilibrado[c(fold[[2]], fold[[3]], fold[[4]]),AR]
    x_entreno<-x_escalado[c(fold[[2]], fold[[3]], fold[[4]]),]
    cat(i, 1, "\n")
  }
  

  set.seed(44)
  acu<-matrix(NA, nrow=0, ncol=4)
  colnames(acu)<-c("c","gama", "intercept", "acc")
  for (c in 2^(0:3)){
    for (gama in seq(0.001, 0.01,0.001)){ #0.01 en vez de 0.003
      for (intercept in seq(-0.6, 0.6, 0.2)){ #0.6
        
        qdebien<-svm(x=x_entreno, y=y_entreno, scale=scalar, kernel="sigmoid",type="C-classification",cost=c, gamma=gama, coef0=intercept)
        predictingcontent<-predict(qdebien, x_val)
        acu<-rbind(acu, c(c, gama, intercept, mean(predictingcontent == y_val)))
      }
    }
  }
  cos<-acu[which.max(acu[,4]),1]  
  gamba<-acu[which.max(acu[,4]),2]
  intercepto<-acu[which.max(acu[,4]),3]
  C_fold_sigmoid[i]<-cos
  intercept_fold_sigmoid[i]<-intercepto
  gamma_fold_sigmoid[i]<-gamba
  
  
  
  svm_definitivo<-svm(x=x_entreno, y=y_entreno, scale=scalar, kernel="sigmoid",type="C-classification", cost=cos, gamma=gamba, coef0=intercepto )


  
  

  
  
  pred_test<-predict(svm_definitivo, x_test) #1*(prob_test>=rep(umbral, length(prob_test)))
  
  
  confu_test<-table(y_test, pred_test)
  accuracy_sigmoid[i]=(confu_test[1,1]+confu_test[2,2])/sum(confu_test)
  sensitividad_sigmoid[i]=confu_test[2,2]/sum(confu_test[2,])
  precision_sigmoid[i]=confu_test[2,2]/sum(confu_test[,2])
  
  confu_test<-table(y_test, pred_test)
  accuracy_sigmoid[i]=(confu_test[1,1]+confu_test[2,2])/sum(confu_test)
  sensitividad_sigmoid[i]=confu_test[2,2]/sum(confu_test[2,])
  precision_sigmoid[i]=confu_test[2,2]/sum(confu_test[,2])
  
}

cat(accuracy_sigmoid, mean(accuracy_sigmoid), "\n")
cat(sensitividad_sigmoid, mean(sensitividad_sigmoid), "\n")
cat(precision_sigmoid, mean(precision_sigmoid), "\n")
cat(C_fold_sigmoid, "\n")
cat(gamma_fold_sigmoid, "\n")
cat(intercept_fold_sigmoid, "\n")






















### 6 Regresión logística con selección de variables forward con criterio AIC
### para cada i vamos rotando los conjuntos test (1 fold), entreno (4 fold). 
### Calculamos acc, precision y sensitiv en el conjunto test para cada rotación y imprimimos los vectores asociados y sus medias


accuracy_logistic<-vector(mode = "double", length=5)
precision_logistic<-vector(mode = "double", length=5)
sensitividad_logistic<-vector(mode = "double", length=5)



predic=function(logis, len){
  length(logis)
  logis_predic=rep(0, len) ## prediccion 0: menos criminalidad que la mediana
  logis_predic[logis > 0.5]=1 #y ahora ponemos 1 si la predicción de pi es mayor q 0.5}
  logis_predic
}

for (i in 1:5){
  
  conjunto_test<-dataset_equilibrado[fold[[i]],]
  cat("vamos por el paso", i, "\n")
  
  if (i==4){
    conjunto_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]], fold[[5]]),]
  }
  if (i==3){
    conjunto_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]], fold[[4]]),]
    
  }
  if (i==2){
    conjunto_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]], fold[[3]]),]
  }
  if (i==1){
    conjunto_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]], fold[[2]]),]
  }    
  
  if (i==5){
    conjunto_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[1]], fold[[2]]),]
    
  }
  
  
  set.seed(44)
  
  
  
  null=glm(AR~1, data=conjunto_entreno, family=binomial) #definimos el modelo sin variables
  logis=glm(as.formula(formula), data=conjunto_entreno, family=binomial) #datos_estandarizados_[-test_random,] es la parte de entrenamiento de los datos estandizados 
  logis_bw=stepAIC(null,#logis,
                   scope=list(lower=null, upper=logis), 
                   direction="forward", #"backward",
                   k=2 ,trace=0)
  prob_test=predict(logis_bw,newdata=conjunto_test, type="response", se =T)
  pred_test=predic(prob_test$fit, dim(conjunto_test)[1]) #datos_estandarizados[test_random] datos estandarizados variables predictoras en conjunto test
  
  
  
  
  
  confu_test<-table(conjunto_test[, AR], pred_test)
  accuracy_logistic[i]=(confu_test[1,1]+confu_test[2,2])/sum(confu_test)
  sensitividad_logistic[i]=confu_test[2,2]/sum(confu_test[2,])
  precision_logistic[i]=confu_test[2,2]/sum(confu_test[,2])
  
}

cat(accuracy_logistic, mean(accuracy_logistic), "\n")
cat(sensitividad_logistic, mean(sensitividad_logistic), "\n")
cat(precision_logistic, mean(precision_logistic), "\n")

 