### Script comparación resultados de diversos métodos de aprendizaje automático en detección de ríos atmosféricos.
### El imput es el de las imágenes de persistencia 0 y 1 dimensionales



####1 Carga de librerías y de parámetros

library(readr)
library(data.table)
library(ROSE)
library(class) #Se necesita para knn
library(pROC)
library(e1071)
library(MASS)
library(caret)


###MIRAR SI SE HA PUESTO EL ROWNAME COMO COLUMNA EN LOS READ_csv, EN FUNCION DE ESO ELIMINAR LA PRIMERA COLUMNA O NO EN EL MERGE (PARA AMBOS DATA TABLE)
###TAMBIEN SI SE PUSO EL ROWNAME COMO COLUMNA CAMBIAR LO TRATAR POR SEPAREADO A
#LOS NNOMBRES DE COLUMNA 1,2 POR LOS NOMBRES DE COLUMNA 1,2,3


daily <- 1


#Variables que indican si se escala en cada algoritmo:
# 1o: se calculan medias y varianzas en train y se aplican a train y a validación,
#y 2o :para el modelo final, se hace lo mismo solo que calculando para train+validación
# y aplicandolas para train+validación y test

escalar_KNN <- TRUE
escalar_SVM_RAD<- FALSE 
escalar_SVM_LINEAL<- FALSE #TRUE
escalar_SVM_POLIN<- FALSE 
escalar_SVM_SIGMO<- FALSE 


generar_folds_datos <- function(dataset_equilibrado, x, fold, y_var_name="AR", escalar = FALSE) {
  
  n_folds <- length(fold)
  resultados <- vector("list", n_folds)
  
  for (i in 1:n_folds) {
    # Índices para validación y test
    val_idx  <- fold[[i]]
    test_idx <- fold[[(i %% n_folds) + 1]]  # siguiente fold, vuelve a 1 si es el último
    
    # Índices para entrenamiento = todos menos val_idx y test_idx
    train_folds <- setdiff(seq_len(n_folds), c(i, (i %% n_folds) + 1))
    train_idx <- unlist(fold[train_folds])
    
    # --- Extraer conjuntos usando el nombre de variable como string ---
    y_val <- dataset_equilibrado[val_idx, ][[y_var_name]]
    x_val <- x[val_idx, ]
    
    y_test <- dataset_equilibrado[test_idx, ][[y_var_name]]
    x_test <- x[test_idx, ]
    
    y_entreno <- dataset_equilibrado[train_idx, ][[y_var_name]]
    x_entreno <- x[train_idx, ]
    
    # Escalado opcional
    if (escalar) {
      scaling_params <- preProcess(x_entreno, method = c("center", "scale"))
      x_entreno_preproc <- predict(scaling_params, x_entreno)
      x_val_preproc     <- predict(scaling_params, x_val)
      
      # Guardar resultados
      resultados[[i]] <- list(
        x_train = x_entreno,
        x_train_preproc = x_entreno_preproc,
        y_train = y_entreno,
        x_val   = x_val,
        x_val_preproc = x_val_preproc,
        y_val   = y_val,
        x_test  = x_test,
        y_test  = y_test, 
        scaling_params = scaling_params
      )
    } else {
      x_entreno_preproc <- x_entreno
      x_val_preproc     <- x_val
      
      # Guardar resultados
      resultados[[i]] <- list(
        x_train = x_entreno,
        x_train_preproc = x_entreno_preproc,
        y_train = y_entreno,
        x_val   = x_val,
        x_val_preproc = x_val_preproc,
        y_val   = y_val,
        x_test  = x_test,
        y_test  = y_test
      )
    }
    

    
    # Mostrar el orden de folds usados (opcional)
    cat("Iteración", i, 
        " | Validación:", i, 
        " | Test:", (i %% n_folds) + 1, 
        " | Entreno:", paste(train_folds, collapse = ","), "\n")
  }
  
  return(resultados)
}



preparar_datasets_modelo_validado <- function(x_entreno, y_entreno, x_val, y_val, x_test, escalar = TRUE) {
  
  # Unir entrenamiento y validación
  x_train_and_val <- rbind(x_entreno, x_val)
  y_train_and_val <- c(y_entreno, y_val)
  
  if (escalar) {
    scaling_params <- preProcess(x_train_and_val, method = c("center", "scale"))
    x_train_and_val_preproc <- predict(scaling_params, x_train_and_val)
    x_test_preproc <- predict(scaling_params, x_test)
  } else {
    x_train_and_val_preproc <- x_train_and_val
    x_test_preproc <- x_test
  }
  
  return(list(
    x_train_and_val_preproc = x_train_and_val_preproc,
    x_train_and_val = x_train_and_val,
    y_train_and_val = y_train_and_val,
    x_test = x_test_preproc
  ))
}

##### 2. Carga y tratamiento de datos


if (daily==1){ topological_info<-as.data.table(read_csv("../../cv2_vectors_0.25_daily_until_06_2017_40.25.csv", 
                                              col_types = cols(...1 = col_skip(), ano = col_double(), 
                                                               t = col_double()))) 
}else{
  topological_info<-as.data.table(read_csv("../../cv2_vectors_0.25_until_06_2017_40.25.csv", 
                                 col_types = cols(...1 = col_skip(), ano = col_double(), 
                                                  t = col_double())))
}


if(daily==1) {
  etiquetas<-as.data.table(read_csv("../../etiquetas_merra_2_daily.csv", 
                                    col_types = cols(...1 = col_skip(), ano = col_double(), 
                                                     t = col_double())))
} else {
  etiquetas<-as.data.table(read_csv("../../etiquetas_merra_2.csv", 
                                    col_types = cols(...1 = col_skip(), ano = col_double(), 
                                                     t = col_double())))
}



summary(etiquetas)

etiquetas[,AR:=as.factor(etiquetas[,AR])] # Variable AR la convertimos en factor

#cambiamos nombres variables año e imagen 
colnames(topological_info)[1:2]<-colnames(etiquetas)[1:2]

#cambiamos nombres variables de imágenes de persistencia
colnames(topological_info)[-c(1,2)]<-paste0("X",colnames(topological_info)[-c(1,2)])


#Unimos las etiquetas AR a las imágenes de persistencia equilibradas
dataset_desequilibrado<-merge(x=etiquetas[,], y=topological_info[,], by=colnames(etiquetas)[1:2], all.y=FALSE, all.x=FALSE)

table(dataset_desequilibrado[, AR])



# Resampleamos eliminando datos que tienen AR=0 hasta obtener clase AR=1 el 48% de los datos
dataset_equilibrado <-as.data.table(ovun.sample(AR~., data=dataset_desequilibrado,
                                                p=0.48, 
                                                seed=1, method="under")$data)

rm(topological_info) #eliminamos datatables q ya no son necesarios
gc()
class(dataset_equilibrado)

table(dataset_equilibrado[,AR])
prop.table(table(dataset_equilibrado[,AR]))
prop.table(table(dataset_desequilibrado[,AR]))


# nos quedamos con ciertas columnas como variables explicativas: 
# primero eliminamos las 3 primeras columnas (índices y etiquetado de AR)
x <- dataset_equilibrado[, -(1:3)]

# segundo: Eliminamos las columnas explicativas que tienen varianza 0 
#(sapply calcula la varianza de cada columna numérica)
cols_con_varianza <- sapply(x, function(x) var(x, na.rm = TRUE) != 0)
x <- x[,..cols_con_varianza]

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
#fold


#fijamos semilla única para todos los métodos de ML
set.seed(456)


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


k_fold<-vector(mode = "integer", length=5)


datasets_mod <- generar_folds_datos(
  dataset_equilibrado = dataset_equilibrado,
  x = x,
  fold = fold,
  y_var_name = "AR",
  escalar = escalar_KNN
)


for (i in 1:5){
  
  cat("iteracion", i, "\n")
  
  accuracy_validacion_1<-vector(mode = "double", length=75) ##vector con la acuracy para cada valor de k 
  
  # knn para cada train y para cada k y tomamos métricas de clasificación en validación
  
  for (k in c(1:10, seq(15, 75, 5))){
    
    modeloi<- knn(train=datasets_mod[[i]]$x_train_preproc, test=datasets_mod[[i]]$x_val_preproc, cl=datasets_mod[[i]]$y_train, k = k, use.all=FALSE) 
    accuracy_validacion_1[k]<- mean(modeloi == datasets_mod[[i]]$y_val)
    
  }
  
  cat(accuracy_validacion_1, " \n")
  
  # máximizamos la accuracy en función de k y guardamos
  
  k=which.max(accuracy_validacion_1)
  cat(k, accuracy_validacion_1[k], " \n")
  k_fold[i]<-k
  
  # knn para train y valicación junto
  # y tomamos métricas de clasificación en test con dicha k óptima
  
  
  datasets_mod_valid<-preparar_datasets_modelo_validado(  x_entreno = datasets_mod[[i]]$x_train,
                                                          y_entreno = datasets_mod[[i]]$y_train,
                                                          x_val     = datasets_mod[[i]]$x_val,
                                                          y_val     = datasets_mod[[i]]$y_val,
                                                          x_test    = datasets_mod[[i]]$x_test,
                                                          escalar = escalar_SVM_RAD  )
  
  pred_test<-knn(train=datasets_mod_valid$x_train_and_val_preproc, test=datasets_mod_valid$x_test, cl=datasets_mod_valid$y_train_and_val , k = k,use.all=FALSE, prob=FALSE) 
  
  
  matriz_confusion<-table(datasets_mod[[i]]$y_test, pred_test)

  
  print(matriz_confusion)
  
  accuracy[i]=(matriz_confusion[1,1]+matriz_confusion[2,2])/sum(matriz_confusion)
  sensitividad[i]=matriz_confusion[2,2]/sum(matriz_confusion[2,])
  precision[i]=matriz_confusion[2,2]/sum(matriz_confusion[,2])
}    
cat(k_fold[[1]],k_fold[[2]],k_fold[[3]],k_fold[[4]],k_fold[[5]], "\n")
cat(accuracy, mean(accuracy), "\n")
cat(sensitividad, mean(sensitividad), "\n")
cat(precision, mean(precision), "\n")




                                                            accuracy<-vector(mode = "double", length=5)
                                                            precision<-vector(mode = "double", length=5)
                                                            sensitividad<-vector(mode = "double", length=5)
                                                            
                                                            
                                                            k_fold<-vector(mode = "integer", length=5)
                                                            for (i in 1:5){
                                                              
                                                              conjunto_validacion<-dataset_equilibrado[fold[[i]],]
                                                              x_val<-x[fold[[i]],]
                                                              if (i!=5){
                                                                conjunto_test<-dataset_equilibrado[fold[[i+1]],]
                                                                x_test<-x[fold[[i+1]],]
                                                                if (i==4){
                                                                  conjunto_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]]),]
                                                                  x_entreno<-x[c(fold[[1]], fold[[2]], fold[[3]]),]
                                                                  cat(i, i+1, 1,2,3, "\n")
                                                                }
                                                                if (i==3){
                                                                  conjunto_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]]),]
                                                                  x_entreno<-x[c(fold[[5]], fold[[1]], fold[[2]]),]
                                                                  cat(i, i+1, 5,1,2, "\n")
                                                                }
                                                                if (i==2){
                                                                  conjunto_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]]),]
                                                                  x_entreno<-x[c(fold[[4]], fold[[5]], fold[[1]]),]
                                                                  cat(i, i+1, 4,5,1, "\n")
                                                                }
                                                                if (i==1){
                                                                  conjunto_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]]),]
                                                                  x_entreno<-x[c(fold[[3]], fold[[4]], fold[[5]]),]
                                                                  cat(i, i+1, 3,4,5, "\n")
                                                                }    
                                                              }
                                                              if (i==5){
                                                                conjunto_test<-dataset_equilibrado[fold[[1]],]
                                                                x_test<-x[fold[[1]],]
                                                                conjunto_entreno<-dataset_equilibrado[c(fold[[2]], fold[[3]], fold[[4]]),]
                                                                x_entreno<-x[c(fold[[2]], fold[[3]], fold[[4]]),]
                                                                cat(i, 1, "\n")
                                                              }
                                                              
                                                              
                                                              accuracy_validacion_1<-vector(mode = "double", length=75) ##vector con la acuracy en 
                                                              
                                                              
                                                              y_validacion<-conjunto_validacion[, AR]#==1
                                                              y_entreno<-conjunto_entreno[, AR]#==1
                                                              y_test<-conjunto_test[, AR]#==1
                                                              
                                                              
                                                              ######escalamos (o no) los datos en train
                                                              #y aplicamos el mismo escalado en validación. 
                                                              #Si escalamos nos quitamos las variables con varianza cero
                                                            
                                                            
                                                              
                                                              if (escalar_KNN==TRUE) {
                                                                # Escalar x_train
                                                                scaling_params <- preProcess(x_entreno, method = c("center", "scale"))
                                                                x_train <- predict(scaling_params, x_entreno)
                                                                
                                                                # Aplicar el mismo escalado a x_valid
                                                                x_valid<- predict(scaling_params, x_val)
                                                              }
                                                              else {
                                                                x_train<-x_entreno
                                                                x_valid<-x_val
                                                              }
                                                              
                                                              
                                                              # knn para cada train y para cada k y tomamos métricas de clasificación en validación
                                                              
                                                              for (k in c(1:10, seq(15, 75, 5))){
                                                                
                                                                modeloi<- knn(train=x_train, test=x_valid, cl=y_entreno , k = k, use.all=FALSE) 
                                                                accuracy_validacion_1[k]<- mean(modeloi == y_validacion)
                                                              }
                                                              
                                                              cat(accuracy_validacion_1, " \n")
                                                              
                                                              # máximizamos la accuracy en función de k y guardamos
                                                              
                                                              k=which.max(accuracy_validacion_1)
                                                              cat(k, accuracy_validacion_1[k], " \n")
                                                              
                                                              k_fold[i]<-k
                                                              
                                                              # knn para train y valicación junto
                                                              # y tomamos métricas de clasificación en test con dicha k óptima
                                                              
                                                              #unimos train y validación
                                                            
                                                              conjunto_entreno_and_val <- rbind(conjunto_entreno, conjunto_validacion)  
                                                              
                                                              #Si escalamos nos quitamos las variables con varianza cero
                                                              
                                                              if (escalar_KNN==TRUE) {
                                                                # Escalar x_train_y_vak
                                                                scaling_params <- preProcess(conjunto_entreno_and_val[,-c(1,2,3,4,5,6,7)], method = c("center", "scale"))
                                                                x_train_and_val <- predict(scaling_params, conjunto_entreno_and_val[,-c(1,2,3,4,5,6,7)])
                                                                
                                                                # Aplicar el mismo escalado a x_test
                                                                x_test<- predict(scaling_params, conjunto_test[,-c(1,2,3,4,5,6,7)])
                                                              }
                                                              else {
                                                                
                                                                x_train_and_val<-conjunto_entreno_and_val[,-c(1,2,3)]
                                                                x_test<- conjunto_test[,-c(1,2,3)] 
                                                                
                                                              }
                                                              
                                                              y_train_and_val <- c(y_entreno, y_validacion)
                                                              
                                                              pred_test<-knn(train=x_train_and_val, test=x_test, cl=y_train_and_val , k = k,use.all=FALSE, prob=FALSE) 
                                                              
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
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM

accuracy_radial<-vector(mode = "double", length=5)
precision_radial<-vector(mode = "double", length=5)
sensitividad_radial<-vector(mode = "double", length=5)
C_fold_radial<-vector(mode = "double", length=5)
gamma_fold<-vector(mode = "double", length=5)


datasets_mod <- generar_folds_datos(
  dataset_equilibrado = dataset_equilibrado,
  x = x,
  fold = fold,
  y_var_name = "AR",
  escalar = escalar_SVM_RAD
)


for (i in 1:5){
  
  cat("iteración ", i, "\n")
  acu<-matrix(NA, nrow=0, ncol=3)
  colnames(acu)<-c("gama", "c", "acc")
  for (gama in seq(0.017, 0.033, 0.002)) { #antes seq(0.009, 0.025, 0.002), seq(0.005, 0.019, 0.002) o seq(0.0055, 0.013, 0.0015) 
    for (c in c(4,8,12,18)){ #antes c(1,2,4,8)
      
      qdebien<-svm(x=datasets_mod[[i]]$x_train_preproc, y=datasets_mod[[i]]$y_train, scale=FALSE, type="C-classification", kernel="radial", cost=c, gamma=gama)
      predictingcontent<- predict(qdebien, datasets_mod[[i]]$x_val_preproc) #predict(qdebien, scale(conjunto_validacion[,-c(1,2,3)]))
      acu<-rbind(acu, c(gama, c, mean(predictingcontent == datasets_mod[[i]]$y_val)))
    }
  }
  gamba<-acu[which.max(acu[,3]),1]
  cos<-acu[which.max(acu[,3]),2]  
  
  C_fold_radial[i]<-cos
  gamma_fold[i]<-gamba
  
  

    
  
 datasets_mod_valid<-preparar_datasets_modelo_validado(  x_entreno = datasets_mod[[i]]$x_train,
                                      y_entreno = datasets_mod[[i]]$y_train,
                                      x_val     = datasets_mod[[i]]$x_val,
                                      y_val     = datasets_mod[[i]]$y_val,
                                      x_test    = datasets_mod[[i]]$x_test,
                                      escalar = escalar_SVM_RAD  )

  svm_definitivo<-svm(x=datasets_mod_valid$x_train_and_val_preproc, y=datasets_mod_valid$y_train_and_val, 
                      scale=FALSE, type="C-classification", kernel="radial", cost=cos, gamma=gamba)
  
  
  pred_test<-predict(svm_definitivo, datasets_mod_valid$x_test)
  
  
  
  confu_test<-table(datasets_mod[[i]]$y_test, pred_test)
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
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM


accuracy_lineal<-vector(mode = "double", length=5)
precision_lineal<-vector(mode = "double", length=5)
sensitividad_lineal<-vector(mode = "double", length=5)
C_fold_lineal<-vector(mode = "double", length=5)




datasets_mod <- generar_folds_datos(
  dataset_equilibrado = dataset_equilibrado,
  x = x,
  fold = fold,
  y_var_name = "AR",
  escalar = escalar_SVM_LINEAL
)


for (i in 1:5){
  
  cat("iteración ", i, "\n")
  acu<-matrix(NA, nrow=0, ncol=2)
  colnames(acu)<-c("c", "acc")

  for (c in c(1:10,14,18,23,28,33,39,47,55,64,73,82)){
    
    qdebien<-svm(x=datasets_mod[[i]]$x_train_preproc, y=datasets_mod[[i]]$y_train, scale=FALSE,kernel="linear", type="C-classification",cost=c)
    predictingcontent<-predict(qdebien, datasets_mod[[i]]$x_val_preproc)
    acu<-rbind(acu, c(c, mean(predictingcontent == datasets_mod[[i]]$y_val)))
    
  }
  cos<-acu[which.max(acu[,2]),1]  
  
  C_fold_lineal[i]<-cos
  
  
  datasets_mod_valid<-preparar_datasets_modelo_validado(  x_entreno = datasets_mod[[i]]$x_train,
                                                          y_entreno = datasets_mod[[i]]$y_train,
                                                          x_val     = datasets_mod[[i]]$x_val,
                                                          y_val     = datasets_mod[[i]]$y_val,
                                                          x_test    = datasets_mod[[i]]$x_test,
                                                          escalar = escalar_SVM_LINEAL  )
  
  svm_definitivo<-svm(x=datasets_mod_valid$x_train_and_val_preproc, y=datasets_mod_valid$y_train_and_val, 
                       scale=FALSE,kernel="linear", type="C-classification", cost=cos)
  
  
  
  pred_test<-predict(svm_definitivo, datasets_mod_valid$x_test)
  
  
  confu_test<-table(datasets_mod[[i]]$y_test, pred_test)
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
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM

accuracy_polinomial<-vector(mode = "double", length=5)
precision_polinomial<-vector(mode = "double", length=5)
sensitividad_polinomial<-vector(mode = "double", length=5)
C_fold_polinomial<-vector(mode = "double", length=5)
gamma_fold_polinomial<-vector(mode = "double", length=5)
intercept_fold_polinomial<-vector(mode = "double", length=5)
grado_fold_polinomial<-vector(mode = "double", length=5)


datasets_mod <- generar_folds_datos(
  dataset_equilibrado = dataset_equilibrado,
  x = x,
  fold = fold,
  y_var_name = "AR",
  escalar = escalar_SVM_POLIN
)



for (i in 1:5){
  cat("iteración ", i, "\n")
  acu<-matrix(NA, nrow=0, ncol=5)
  colnames(acu)<-c("c","gama", "intercept", "grado", "acc")
  for (c in c(0.25,0.5, 0.75,1)){ #antes 1,2,4,8 o c(0.25,1,2,4)
    for (gama in seq(0.001, 0.02,0.003)){ #antes seq(0.001, 0.016,0.003) o seq(0.0055, 0.013,0.0015)
      for (intercept in seq(0.3, 0.7, 0.1)){ # antes seq(0, 0.6, 0.2)
        for (grado in 2:3){
          
          qdebien<-svm(x=datasets_mod[[i]]$x_train_preproc, y=datasets_mod[[i]]$y_train, 
                       scale=FALSE,  kernel="polynomial",type="C-classification",
                       cost=c, gamma=gama, coef0=intercept,  degree=grado)
          
          predictingcontent<-predict(qdebien, datasets_mod[[i]]$x_val_preproc)
          acu<-rbind(acu, c(c, gama, intercept, grado, mean(predictingcontent == datasets_mod[[i]]$y_val)))
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
  
  
  
  datasets_mod_valid<-preparar_datasets_modelo_validado(  x_entreno = datasets_mod[[i]]$x_train,
                                                          y_entreno = datasets_mod[[i]]$y_train,
                                                          x_val     = datasets_mod[[i]]$x_val,
                                                          y_val     = datasets_mod[[i]]$y_val,
                                                          x_test    = datasets_mod[[i]]$x_test,
                                                          escalar = escalar_SVM_POLIN  )
  
  svm_definitivo<-svm(x=datasets_mod_valid$x_train_and_val_preproc, y=datasets_mod_valid$y_train_and_val, 
                      scale=FALSE, kernel="polynomial",type="C-classification", cost=cos,
                      gamma=gamba, coef0=intercepto , degree=grao )
  
  pred_test<-predict(svm_definitivo, datasets_mod_valid$x_test)
  
  confu_test<-table(datasets_mod[[i]]$y_test, pred_test)

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
###  En la definicion de x_scalado vemos si escalamos los datos antes de meterlos en la SVM
                                                      
accuracy_sigmoid<-vector(mode = "double", length=5)
precision_sigmoid<-vector(mode = "double", length=5)
sensitividad_sigmoid<-vector(mode = "double", length=5)
C_fold_sigmoid<-vector(mode = "double", length=5)
gamma_fold_sigmoid<-vector(mode = "double", length=5)
intercept_fold_sigmoid<-vector(mode = "double", length=5)

datasets_mod <- generar_folds_datos(
  dataset_equilibrado = dataset_equilibrado,
  x = x,
  fold = fold,
  y_var_name = "AR",
  escalar = escalar_SVM_SIGMO
)



for (i in 1:5){
  cat("iteración ", i, "\n")
  acu<-matrix(NA, nrow=0, ncol=4)
  colnames(acu)<-c("c","gama", "intercept", "acc")
  for (c in 2^(0:3)){ #antes entre potencias -1 y 2 o entre potencias -2 y 1
    for (gama in seq(0.001, 0.01,0.001)){ #antes seq(0.0001, 0.001,0.0001),  seq(0.0005, 0.001,0.0001) o seq(0.001, 0.01,0.001) o ;0.01 en vez de 0.003
      for (intercept in seq(-0.6, 0.6, 0.2)){ #antes seq(-1.0, 0.2, 0.2) seq(-1.4, -0.2, 0.2) o seq(-0.6, 0.6, 0.2) entre +-0.6
        
        
        qdebien<-svm(x=datasets_mod[[i]]$x_train_preproc, y=datasets_mod[[i]]$y_train, 
                     scale=FALSE, kernel="sigmoid",type="C-classification",
                     cost=c, gamma=gama, coef0=intercept)
        
        
        predictingcontent<-predict(qdebien, datasets_mod[[i]]$x_val_preproc)
        acu<-rbind(acu, c(c, gama, intercept,mean(predictingcontent == datasets_mod[[i]]$y_val)))
      }
    }
  }
  cos<-acu[which.max(acu[,4]),1]  
  gamba<-acu[which.max(acu[,4]),2]
  intercepto<-acu[which.max(acu[,4]),3]
  C_fold_sigmoid[i]<-cos
  intercept_fold_sigmoid[i]<-intercepto
  gamma_fold_sigmoid[i]<-gamba
  
  
  
  
  datasets_mod_valid<-preparar_datasets_modelo_validado(  x_entreno = datasets_mod[[i]]$x_train,
                                                          y_entreno = datasets_mod[[i]]$y_train,
                                                          x_val     = datasets_mod[[i]]$x_val,
                                                          y_val     = datasets_mod[[i]]$y_val,
                                                          x_test    = datasets_mod[[i]]$x_test,
                                                          escalar = escalar_SVM_SIGMO  )
  
  svm_definitivo<-svm(x=datasets_mod_valid$x_train_and_val_preproc, y=datasets_mod_valid$y_train_and_val, 
                      scale=FALSE, kernel="sigmoid",type="C-classification", cost=cos,
                      gamma=gamba, coef0=intercepto )
  
  
  pred_test<-predict(svm_definitivo, datasets_mod_valid$x_test)
  
  confu_test<-table(datasets_mod[[i]]$y_test, pred_test)
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

                                                            
                                                            
                                                            
                                                            accuracy_sigmoid<-vector(mode = "double", length=5)
                                                            precision_sigmoid<-vector(mode = "double", length=5)
                                                            sensitividad_sigmoid<-vector(mode = "double", length=5)
                                                            C_fold_sigmoid<-vector(mode = "double", length=5)
                                                            gamma_fold_sigmoid<-vector(mode = "double", length=5)
                                                            intercept_fold_sigmoid<-vector(mode = "double", length=5)
                                                            
                                                            
                                                            
                                                            for (i in 1:5){
                                                              
                                                              y_val<-dataset_equilibrado[fold[[i]],AR]
                                                              x_val<-x[fold[[i]],]
                                                              if (i!=5){
                                                                y_test<-dataset_equilibrado[fold[[i+1]],AR]
                                                                x_test<-x[fold[[i+1]],]
                                                                if (i==4){
                                                                  y_entreno<-dataset_equilibrado[c(fold[[1]], fold[[2]], fold[[3]]),AR]
                                                                  x_entreno<-x[c(fold[[1]], fold[[2]], fold[[3]]),]
                                                                  cat(i, i+1, 1,2,3, "\n")
                                                                }
                                                                if (i==3){
                                                                  x_entreno<-x[c(fold[[5]], fold[[1]], fold[[2]]),]
                                                                  y_entreno<-dataset_equilibrado[c(fold[[5]], fold[[1]], fold[[2]]),AR]
                                                                  
                                                                  cat(i, i+1, 5,1,2, "\n")
                                                                }
                                                                if (i==2){
                                                                  y_entreno<-dataset_equilibrado[c(fold[[4]], fold[[5]], fold[[1]]),AR]
                                                                  x_entreno<-x[c(fold[[4]], fold[[5]], fold[[1]]),]
                                                                  cat(i, i+1, 4,5,1, "\n")
                                                                }
                                                                if (i==1){
                                                                  y_entreno<-dataset_equilibrado[c(fold[[3]], fold[[4]], fold[[5]]),AR]
                                                                  x_entreno<-x[c(fold[[3]], fold[[4]], fold[[5]]),]
                                                                  cat(i, i+1, 3,4,5, "\n")
                                                                }    
                                                              }
                                                              if (i==5){
                                                                y_test<-dataset_equilibrado[fold[[1]],AR]
                                                                x_test<-x[fold[[1]],]
                                                                y_entreno<-dataset_equilibrado[c(fold[[2]], fold[[3]], fold[[4]]),AR]
                                                                x_entreno<-x[c(fold[[2]], fold[[3]], fold[[4]]),]
                                                                cat(i, 1, "\n")
                                                              }
                                                              
                                                              
                                                              if (escalar_SVM_SIGMO==TRUE) {
                                                                # Escalar x_train
                                                                scaling_params <- preProcess(x_entreno, method = c("center", "scale"))
                                                                x_entreno_preproc <- predict(scaling_params, x_entreno)
                                                                
                                                                # Aplicar el mismo escalado a x_valid
                                                                x_val_preproc<- predict(scaling_params, x_val)
                                                              }
                                                              else {
                                                                x_entreno_preproc<-x_entreno
                                                                x_val_preproc<-x_val
                                                              }
                                                              
                                                              acu<-matrix(NA, nrow=0, ncol=4)
                                                              colnames(acu)<-c("c","gama", "intercept", "acc")
                                                              for (c in 2^(0:3)){
                                                                for (gama in seq(0.001, 0.01,0.001)){ #0.01 en vez de 0.003
                                                                  for (intercept in seq(-0.6, 0.6, 0.2)){ #0.6
                                                                    
                                                                    qdebien<-svm(x=x_entreno_preproc, y=y_entreno, scale=FALSE, kernel="sigmoid",type="C-classification",cost=c, gamma=gama, coef0=intercept)
                                                                    predictingcontent<-predict(qdebien, x_val_preproc)
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
                                                              
                                                              
                                                              
                                                              x_train_and_val <- rbind(x_entreno, x_val)  
                                                              y_train_and_val <- c(y_entreno, y_val)
                                                              
                                                              if (escalar_SVM_SIGMO==TRUE) {
                                                                # Escalar x_train
                                                                scaling_params <- preProcess(x_train_and_val, method = c("center", "scale"))
                                                                x_train_and_val_preproc <- predict(scaling_params, x_train_and_val)
                                                                
                                                                # Aplicar el mismo escalado a x_TEST
                                                                x_test_preproc<- predict(scaling_params, x_test)
                                                              }
                                                              else {
                                                                x_train_and_val_preproc<-x_train_and_val
                                                                x_test_preproc<-x_test
                                                              }
                                                              
                                                              
                                                              svm_definitivo<-svm(x=x_train_and_val_preproc, y=y_train_and_val, scale=FALSE, kernel="sigmoid",type="C-classification", cost=cos, gamma=gamba, coef0=intercepto )
                                                              
                                                              pred_test<-predict(svm_definitivo, x_test_preproc) #1*(prob_test>=rep(umbral, length(prob_test)))
                                                              
                                                              
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


#funcion que pasa de un vector de probabildades a uno de predicciones binarias en función de un umbral en la probabilidad

predic=function(logis, len, threshold){ 
  logis_predic=rep(0, len) ## prediccion 0: menos criminalidad que la mediana
  logis_predic[logis > threshold]=1 #y ahora ponemos 1 si la predicción de pi es mayor q el corte}
  logis_predic
}

#formula del modelo con todas las variables explicativas
formula<-paste( "AR", paste0(colnames(x) , collapse = ' + '), sep = ' ~ ')



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
  
  

  
  
  null=glm(AR~1, data=conjunto_entreno, family=binomial) #definimos el modelo sin variables
  logis=glm(as.formula(formula), data=conjunto_entreno, family=binomial) #datos_estandarizados_[-test_random,] es la parte de entrenamiento de los datos estandizados 
  logis_bw=stepAIC(null,#logis,
                   scope=list(lower=null, upper=logis), 
                   direction="forward", #"backward",
                   k=2 ,trace=0)
  prob_test=predict(logis_bw,newdata=conjunto_test, type="response", se =T)
  pred_test=predic(prob_test$fit, dim(conjunto_test)[1], threshold=0.5) #datos_estandarizados[test_random] datos estandarizados variables predictoras en conjunto test
  
  
  
  
  
  confu_test<-table(conjunto_test[, AR], pred_test)
  accuracy_logistic[i]=(confu_test[1,1]+confu_test[2,2])/sum(confu_test)
  sensitividad_logistic[i]=confu_test[2,2]/sum(confu_test[2,])
  precision_logistic[i]=confu_test[2,2]/sum(confu_test[,2])
  
}

cat(accuracy_logistic, mean(accuracy_logistic), "\n")
cat(sensitividad_logistic, mean(sensitividad_logistic), "\n")
cat(precision_logistic, mean(precision_logistic), "\n")

