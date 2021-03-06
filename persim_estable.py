#######   1- Carga de librerías

#from email.headerregistry import ContentTypeHeader # no se usa
from netCDF4 import Dataset as readnc
import gudhi as gd  
import numpy as np
import pandas as pd
import cv2
import math
import numpy.ma as ma
import scipy.misc 
import itertools as it
from persim import PersistenceImager
import matplotlib.pyplot as plt
import gc


#######   2- Parametros

#primer año, mes, día y hora
an=1980
di="01"
me="01"
hor="00"

nomes=["ano", "t"]
pivote1=47.8
pivote2=40.5
pivote3=22.8
lat_min=19
lat_max=56
lon_min=-179
lon_max=-109
daily=0 #vale 1 si consideramos solo datos con precisión diaria y no 3-horaria (cojemos los datos de las imágenes 0,8,16, ... corespondientes a las 00)

pixel_size_dimcero=5#tamaño del pixel de las imágenes de persistencia de dimensión 0
pixel_size_dimuno=4#tamaño del pixel de las imágenes de persistencia de dimensión 1






#######  3- Calculamos las coordenadas de la región de estudio y las subregiones de interés para descartar componentes conexas 

direccion="E:"'\\'+str(an)+"\ARTMIP_MERRA_2D_"+str(an)+me+di+"_"+hor+".nc"
print(direccion)

ncfile = readnc(direccion, "r") #leemos los datos

lat=ncfile.variables["lat"][:] #vector de latitudes
la=(lat>=lat_min)&(lat<=lat_max) #nos quedamos con las latitudes que nos interesan (las de la región del estudio)
    

lon=ncfile.variables["lon"][:]#vector de longitudes

lo=(lon>=lon_min)&(lon<=lon_max) #nos quedamos con las longitudes que nos interesan (las de la región del estudio)
  
lats_usadas=ma.getdata(lat[la]) #filtamos latitudes
lons_usadas=ma.getdata(lon[lo]) #filtramos longitudes
#latlon=((lats_usadas>38.9+0*lons_usadas))

lons_america=[[]] * lats_usadas.shape[0] #creamos una lista de listas, una por cada latitud, que nos indica, para cada latitud que longitudes corresponden con el continente americano

lathawai=(lat[la]>=lat_min)&(lat[la]<22.8) #latitudes de hawwai

for i in np.arange(0,lats_usadas.shape[0] , 1):
    if lats_usadas[i]>pivote2:
        if lats_usadas[i]<= pivote1:
            lons_america[i]=(lons_usadas>-125.5)
        else:
           lons_america[i]=(lons_usadas>-1.411*lats_usadas[i]-58.465)
    else: 
        if lats_usadas[i]>=pivote3:
            lons_america[i]=(lons_usadas>-0.758*lats_usadas[i]-95.1132)
        else:
           lons_america[i]=(lons_usadas>-109)

lons_america=[[]] * lats_usadas.shape[0]  #lista donde para cada indice de latitudes que usamos ponemos un array donde es TRUE las longitudes que se corresponden con America





###### 4- Definimos funciones


def myfunc(a, b): 
    return a in b
invect=np.vectorize(myfunc) #funcion que nos da una matriz booleana donde TRUE implica que el elemento a de la matriz imput está en el conjunto b

def get_key(val, dicc):
    for key, value in dicc.items():
         if val == value:
             return key
 
    return "There is no such Key"    ##función que nos da la clave de un elemento de un diccionario

####### 5- Definimos variables de tamaño y vectores para ciertos grupos. Inicializamos listas y valores.

#print(type(tmq))
longlat=lats_usadas.shape[0] #número de latitudes y de longitudes
longlon=lons_usadas.shape[0]
print("dimensiones array=" , longlat, longlon)



xval = np.arange(0,longlat , 1) #la x en realidad es por ser la primera dimensión pero realmente es "abajo-arriba", representa las filas de una matriz
yval = np.arange(0,longlon , 1) #la y en realidad es por ser la segunda dimensión pero realmente es "izquierda-derecha", representa las columnas de una matriz
contadorcambio=0
contadorcambiomedia=0 #contadorcambio y contadormedia es para medir el efecto de los cambios en la fillterfun por seleccionar comp. conexas. 
#No es imprescindible este conteo. Tal como está el código no se usa, está comentado. Habría finalmente que dividir los contadores entre nfila para ver media por imagen

#inicializamos listas 
persis0=[] #lista dónde por cada imagen tenemos sus features 0 dim (quitando la primera, la que tiene persistencia infinita)
persis1=[] #lista dónde por cada imagen tenemos sus features 1 dim


if daily==1:
    horasvalue=np.array(['00'])
else:
    horasvalue=np.array(['00', '03', '06', '09', '12', '15', '18', '21'])
diccionario={} ##diccionario que para cada año nos va a decir que número de imagen se corresponde con la primera imagen del año


nfila=0 ##contador del número de imágenes ya computadas



########6- Bucle de computación de persistencia. Leemos la matriz de datos para cada año, mes, día y hora. Para cada ano y para cada mes el número de días del mes varía.
##Primero modificamos la filterfun para que los conjuntos de nivel superior no contengan componentes conexas que no intersequen ni la cossta americana ni las latitudes de hawai
# # y luego calculamos el complejo cúbico y sus persitencias 0 y 1 dimensionales

for ano in np.arange(1980,2018,1):



    im=0
    diccionario[ano]=nfila
    print(diccionario[ano])

    if ano==2017:
        mesesvalue=np.array([ '01', '02', '03', '04', '05', '06'])
    else:
        mesesvalue=np.array([ '01', '02', '03', '04', '05', '06',  '07', '08', '09', '10', '11', '12'])



    for mes in mesesvalue:
        print("empezamos carga e persitencia do mes "+ mes+ " do ano", ano )

        if mes in {"01", "03", "05", "07", "08", "10", "12"}: #meses con 31 días
            diasvalue=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29','30', '31'])
        if mes in {"04", "06", "09","11"}: #meses con 30 días
            diasvalue=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29','30'])
        if (mes=="02")&(ano in {1980,1984,1988,1992,1996,2000, 2004,2008,2012,2016}): #año bisiesto febrero tiene 29 días
            diasvalue=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29'])
            print(ano, "foi bisiesto")        
        if (mes=="02")&(ano not in {1980,1984,1988,1992,1996,2000, 2004,2008,2012,2016}): #año no bisiesto febrero tiene 28 días
            diasvalue=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28'])
        
        for dia in diasvalue:
            for hora in horasvalue:
                direccion="E:"'\\'+str(ano)+"\ARTMIP_MERRA_2D_"+str(ano)+mes+dia+"_"+hora+".nc"
                ncfile = readnc(direccion, "r") #leemos los datos de esa imagen
                iwp=ncfile.variables["IWV"][la,lo] ##cargamos la matriz de IWV en la región del estudio
                im=im+1
                nfila=nfila+1
                #primero modificamos la filterfun para quitarnos ccs que no intersequen las regiones de interés (latitudes de hawai y continente americano)

                filterfun=-iwp
                conxuntocero={0}
                indicador=np.zeros((iwp.shape[0], iwp.shape[1]))

                for l in np.arange(60,- 1, -0.25): #bucle por umbral donde vamos a calcular las ccs del superlevelset asociado al umbral
                    umbral=-l
                    superlevel=np.uint8(np.array(iwp>l, dtype=np.int16))
                    ccs=cv2.connectedComponents(superlevel)[1]
                    ccshawai=ccs[lathawai, :]  #cojemos las ccs que intersecan con las latitudes de hawai  
                    listadeccs=[ccshawai[u, v] for u in np.arange(0,ccshawai.shape[0] , 1) for v in np.arange(0,ccshawai.shape[1] , 1)] 
                    #inicializamos la lista con las ccs que intersecan con las latitudes de hawai
                    
                    for k in np.arange(0,lats_usadas.shape[0] , 1): #bucle por latitud dónde añadimos a la lista las ccs que intersecan con América (en cada latitud)
                        componentes=ccs[k, lons_america[k]] #array donde seleccionamos las ccs de la costa americana para la latitud k
                        listadeccs=listadeccs+componentes.tolist() #las añadimos a la lista
                    ccsinteres=set(listadeccs) #pasamos la lista a conjunto (eliminando las repetidas)
                    
                    
                    boolean=invect(ccs, ccsinteres-conxuntocero) #matriz que vale TRUE si ese i,j pertenece a una cc

                    indicador=np.where(boolean, indicador+1, indicador) #esta matriz tiene un valor 0 si en ningún momento boolean fue true hasta ahora
                    #filterfun=np.where(boolean, np.where(indicador==1, umbral, filterfun) , umbral)
                    filterfun= np.where(indicador==1, umbral, filterfun)
                    #cambiamos la filterfun para que no entren ccs en los superlevelsets que no intersequen las regiones de interés (america, hawai) en este umbral 
                    #de esta manera reducimos algún valor de la función en la rejilla hasta que no esté en una cc en los superlevelsets que no intersequen las regiones de interés (america, hawai)
                    #cambiamos signo xq gudhi solo considera sublevel y no superlevel sets
                
                #Con este contador medimos cuanto cambia en media y máximo los cambios en la filterfun respecto de los valores iwp originales. Queda comentado pero se puede descomentar:
                #contadorcambio=contadorcambio+np.amax(iwp+filterfun)
                #contadorcambiomedia=contadorcambiomedia+np.mean(iwp+filterfun) #sumamos a los contadores la media y maximo de los cambios



                filterfun_list = np.array([filterfun[u, v] for u in xval for v in yval]) #función de filtrado (lista con el IWV (modificado en el anterior bucle)  siguiendo orden lexicográfico de coordenadas)
                cc_river = gd.CubicalComplex( dimensions = [longlat ,longlon], 
                        top_dimensional_cells = filterfun_list
                        ) #construcción del complejo cúbico
                    
                

                persi=cc_river.persistence()    #guardamos la peristencia     
                    
                
                filtro0 = filter(lambda featur: featur[0] == 0, persi)

                persi0=list(filtro0) #lista con los intervalos de dim 0 y su dim (0 siempre)
                persi0.pop(0)
                persi0=[featur[1] for featur in persi0] #lista con los intervalos de persistencia de dim 0 de la imagen (menos el primero)
                persis0.append(persi0)
                #edad_muerte_0=list(map(tiempovida, persi0)) #lista con la longitud de intervalos de persistencia de dim 0 de la imagen (menos el primero)
                #edades_muerte_0.append(edad_muerte_0)


                filtro1 = filter(lambda featur: featur[0] == 1, persi)
                persi1=list(filtro1) #lista con los intervalos de dim 0 y su dim (1 siempre)
                persi1=[featur[1] for featur in persi1] #lista con los intervalos de persistencia de dim 1 de la imagen 
                persis1.append(persi1)
                #edad_muerte_1=list(map(tiempovida, persi1)) #lista con la longitud de intervalos de persistencia de dim 0 de la imagen
                #edades_muerte_1.append(edad_muerte_1)
    
    print("en ano", ano, "hubo ", im, "imágenes")
print("hay un total de ", nfila, "imágenes")


#### 7- Eliminamos datos no necesarios

del iwp
del filterfun
del boolean
del indicador
del cc_river
del persi
gc.collect()



#######8- CALCULAMOS LAS IMÁGENES DE PERSISTENCIA DE DIM=1



pimgr = PersistenceImager(pixel_size=pixel_size_dimuno) #definimos la clase imagen de persistencia y ponemos una precisión

pdgms=[] #lista donde para cada imagen obtenemos un array con los intervalos de persistencias (los intervalos de persistencias son listas y no tuples)
for im in np.arange(0, nfila, 1): 
    lst=[[featur[0],featur[1]] for featur in persis1[im]]
    pdgms.append(np.asarray(lst))


pimgr.fit(pdgms, skew=True) #ajustamos el tamaño de imagen al rango máximo en la lista de las persistencias
print(pimgr)
tamaño_persis=pimgr.resolution ##vemos el tamaño de la imagen de persistencia

print(tamaño_persis)

del persis1 #borramos datos no necesarios
gc.collect()

pimgs = pimgr.transform(pdgms, skew=True, n_jobs=-1) #lista donde tenemos para cada im un array con los valores de la densidad en cada rejilla
data = [] #lista donde para cada imagen añadiremos un vector con año, imagen, persimdim1 y persimdim0 aplanadas




for im in np.arange(0, nfila, 1):  ##para cada im añadimos en la lista data el año correspondiente, el número de la imagen en el año y la imagen de persistencia de dim 1 aplanada 
    xval = np.arange(0,tamaño_persis[0] , 1)
    yval = np.arange(0,tamaño_persis[1] , 1)
    valores_dicc=list(diccionario.values()) #calculamos el año en el que está la imagen a través de diccionario (tiene como llaves los años y el índice de la primera imagen )
    if im not in set(valores_dicc):#si el índice de la imagen no es un valor (o elemento) del diccionario (i.e. no es la primera imagen de un año) entonces el año en el que estamos es la llave de la imagen justo menor a ella que esté en los values del diccionario
        valores_dicc.append(im)
        valores_dicc.sort()
        ano=get_key(valores_dicc[valores_dicc.index(im)-1], diccionario)  ##get_key función que nos da la clave de un elemento de un diccionario

    else: #si el índice de la imagen es un valor (o elemento) del diccionario (i.e. es la primera imagen de un año) entonces el año en el que estamos es la llave de ese valor
        ano=get_key(valores_dicc[valores_dicc.index(im)],diccionario)  ## get_key función que nos da la clave de un elemento de un diccionario
        print("comezamos a calcular imaxes de persistencia de dim 1 para o ano", ano)

    imaxe_do_ano=im-diccionario[ano] #calculamos que número de imagen dentro de un año es im una vez ya sabemos el año y con diccionario[ano] el índice de la primera imagen de ese año
    
    if daily==1:
        data.append([ano,imaxe_do_ano*8]+[pimgs[im][u, v] for u in xval for v in yval]) #añadimos el array como lista aplanada a data
    else: 
        data.append([ano,imaxe_do_ano]+[pimgs[im][u, v] for u in xval for v in yval]) 

    

pdgms=[]#borro la memoria ya que vamos a reiniciar con dim 1
pimgs=[]


#######9- CALCULAMOS LAS IMÁGENES DE PERSISTENCIA DE DIM=0
pimgr = PersistenceImager(pixel_size=pixel_size_dimcero) #definimos la clase imagen de persistencia y ponemos una precisión
pdgms=[]  #lista donde para cada imagen obtenemos un array con los intervalos de persistencias (los intervalos de persistencias son listas y no tuples)
for im in np.arange(0, nfila, 1): 
    lst=[[featur[0],featur[1]] for featur in persis0[im]]
    pdgms.append(np.asarray(lst))

#print(pdgms, type(pdgms))
pimgr.fit(pdgms, skew=True)#ajustamos el tamaño de imagen al rango máximo en la lista de las persistencias
print(pimgr) 
tamaño_persis=pimgr.resolution 

print(tamaño_persis)

del persis0 #borramos datos no necesarios
gc.collect()

pimgs = pimgr.transform(pdgms, skew=True, n_jobs=-1)  #lista donde tenemos para cada im un array con los valores de la densidad en cada rejilla

for im in np.arange(0, nfila, 1): #añadimos a la fila im de data la imagen de persistencia de dimensión 0 aplanada correspondiente
    xval = np.arange(0,tamaño_persis[0] , 1)
    yval = np.arange(0,tamaño_persis[1] , 1)
    if im % 400==0:
        print("vamos por la imagen", im, "haciendo su imagen de persistencia de dim 0")
    data[im]=data[im]+[pimgs[im][u, v] for u in xval for v in yval] #añadimos el array como lista aplanada a data[im] sumandola a la anterior

del pdgms #borramos datos no necesarios
del pimgs
gc.collect()

####### 10- Guardamos resultados
x=pd.DataFrame(data) #hacemos data frame la lista data (de la vectorización de las persims)
print("dataframe con imaxes de persistencia feito")
if daily==1:
    x.to_csv("C:/Users/David/OneDrive/Documents/mestrado/tfm-solicitude bolsa/piton/persim_estable_daily"+str(pixel_size_dimuno)+str(pixel_size_dimcero)+".csv")
else:
    x.to_csv("C:/Users/David/OneDrive/Documents/mestrado/tfm-solicitude bolsa/piton/persim_estable"+str(pixel_size_dimuno)+str(pixel_size_dimcero)+".csv")



print("gardado como csv")
del x
gc.collect()

