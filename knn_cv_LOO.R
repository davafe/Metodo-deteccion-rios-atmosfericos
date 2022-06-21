### Script KNN CV en detección de ríos atmosféricos.
### El imput es el de las imágenes de persistencia 0 y 1 dimensionales.



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



persim<-as.data.table(read_csv("mestrado/tfm-solicitude bolsa/piton/persim_estable_daily45.csv", 
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
#persim<-persim[, 138:257] #solo dim 0


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


##### 3 Dividimos aleatoriamente en conjunto test y conjunto entrenamiento. 
# Creamos 10 folds que subdividen el conjunto de entrenamiento para CV

nfilas_equilibrado=dim(dataset_equilibrado)[1]
indices<-1:nfilas_equilibrado

set.seed(2)

indices_test<-sample(indices, size=0.25*nfilas_equilibrado)

conj_test<-dataset_equilibrado[indices_test,]
conj_entreno<-dataset_equilibrado[-indices_test,]

nfilas_entreno=dim(conj_entreno)[1]
indices_entreno<-1:nfilas_entreno

permutacion_indices<-sample(indices_entreno, size=nfilas_entreno)

fold<-vector('list',length = 10)
fold[[1]]<-permutacion_indices[1:floor(0.1*nfilas_entreno)]
fold[[2]]<-permutacion_indices[((floor(0.1*nfilas_entreno))+1):(2*floor(0.1*nfilas_entreno))]
fold[[3]]<-permutacion_indices[((2*floor(0.1*nfilas_entreno))+1):(3*floor(0.1*nfilas_entreno))]
fold[[4]]<-permutacion_indices[((3*floor(0.1*nfilas_entreno))+1):(4*floor(0.1*nfilas_entreno))]
fold[[5]]<-permutacion_indices[((4*floor(0.1*nfilas_entreno))+1):(5*floor(0.1*nfilas_entreno))]
fold[[6]]<-permutacion_indices[((5*floor(0.1*nfilas_entreno))+1):(6*floor(0.1*nfilas_entreno))]
fold[[7]]<-permutacion_indices[((6*floor(0.1*nfilas_entreno))+1):(7*floor(0.1*nfilas_entreno))]
fold[[8]]<-permutacion_indices[((7*floor(0.1*nfilas_entreno))+1):(8*floor(0.1*nfilas_entreno))]
fold[[9]]<-permutacion_indices[((8*floor(0.1*nfilas_entreno))+1):(9*floor(0.1*nfilas_entreno))]
fold[[10]]<-permutacion_indices[((9*floor(0.1*nfilas_entreno))+1):nfilas_entreno]
fold

####  4 KNN 
### para cada i vamos rotando el conjunto de validación (1 fold) y entreno (9 fold)
## Entrenamos el modelo en el conjunto entreno y calculamos para diferentes valores de los hiperparámetros 
### la accuracy en el conjunto de validación. Nos quedamos con los valores de hiperparámetros que maximizan esa
##accuracy en el conjunto de entrenamiento.  
###Luego calculamos accuracy, precision y sensitividad en único conjunto test.
## # Finalmente imprimimos estos valores de acc, prec y sens y sus medias.
### Hiperparámetros: k \in \{1,2,..., 75\}

accuracy_cv<-matrix(nrow = 11, ncol = 75)

for (i in 1:10){
  
  conjunto_entreno<-conj_entreno[-fold[[i]],] #conjunto entreno dentro de la CV en esta interación
  
  conjunto_validacion<-conj_entreno[fold[[i]],] #conjunto validación dentro de la CV en esta interación
  
  cat("iteración número", i, "\n")

  ######escalamos (o no) los datos por separado (o no)
  datos_t<-conjunto_entreno[,-c(1,2,3)] 
  datos_v<-conjunto_validacion[,-c(1,2,3)]

  
  y_validacion<-conjunto_validacion[, AR]==1
  y_entreno<-conjunto_entreno[, AR]==1
 
  
  set.seed(456)
  for (k in c(1:75)){
    cat("k número", k, "\n")
    
    modeloi<- knn(train=datos_t, test=datos_v,cl=y_entreno , k = k,use.all=FALSE) 
    accuracy_cv[i,k]<- mean(modeloi == y_validacion)
  }
  
  cat(accuracy_cv[i,], " \n")
} 


accuracy_cv[11,]<-apply(accuracy_cv[-11,], 2, mean)
print(accuracy_cv[11,])
k=which.max(accuracy_cv[11,])
cat(k, accuracy_cv[11,k], " \n")
  
datos_t<-conj_entreno[,-c(1,2,3)] 
y_entreno<-conj_entreno[, AR]==1
datos_test<- conj_test[,-c(1,2,3)] 
y_test<-conj_test[, AR]==1
  

pred_test<-knn(train=datos_t, test=datos_test,cl=y_entreno , k = k,use.all=FALSE, prob=FALSE) 
conj_test<-conj_test[, clasificacion:=pred_test]

matriz_confusion<-table(y_test, pred_test) 

print(matriz_confusion)



accuracy_test=(matriz_confusion[1,1]+matriz_confusion[2,2])/sum(matriz_confusion)
sensitividad_test=matriz_confusion[2,2]/sum(matriz_confusion[2,])
precision_test=matriz_confusion[2,2]/sum(matriz_confusion[,2])

cat(accuracy_test,  "\n")
cat(sensitividad_test, "\n")
cat(precision_test, "\n")

pred_entre<-knn(train=datos_t, test=datos_t,cl=y_entreno , k = k,use.all=FALSE, prob=FALSE) 
conj_entre<-conj_entreno[, clasificacion:=pred_entre]

matriz_confusion_entre<-table(y_entreno, pred_entre) 

print(matriz_confusion_entre)

accuracy_entre=(matriz_confusion_entre[1,1]+matriz_confusion_entre[2,2])/sum(matriz_confusion_entre)
sensitividad_entre=matriz_confusion_entre[2,2]/sum(matriz_confusion_entre[2,])
precision_entre=matriz_confusion_entre[2,2]/sum(matriz_confusion_entre[,2])

cat(accuracy_entre,  "\n")
cat(sensitividad_entre, "\n")
cat(precision_entre, "\n")









