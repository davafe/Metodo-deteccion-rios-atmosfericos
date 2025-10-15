#######   1- Carga de librerías

#from email.headerregistry import ContentTypeHeader # no se usa
from netCDF4 import Dataset as readnc
import gudhi as gd  
import numpy as np
import pandas as pd
import cv2
import numpy.ma as ma
import itertools as it
from persim import PersistenceImager
import matplotlib.pyplot as plt
import gc

####### 1.1 - Importing local modules

from local_functions.elder_conected_components import *
from local_functions.matrix import *

#######   2- Parametros

#primer año, mes, día y hora
an=1980
di="01"
me="01"
hor="00"

nomes=["ano", "t"]
pivot1=47.8
pivot2=40.25
pivot3=22.8
lat_min=19
lat_max=56
lon_min=-179
lon_max=-109
daily=1 #vale 1 si consideramos solo datos con precisión diaria y no 3-horaria (cojemos los datos de las imágenes 0,8,16, ... corespondientes a las 00)
bug=0 #vale 1 si queremos compararnos con el bug detectado en persim_estable del paper, otro valor es el caso contrario
if bug==1:
    bug_text="bug"
else: 
    bug_text=""

proof = 0 # vale 1 si hacemos prueba con muchos menos imágenes

p = 0.25 # paso en la función topológica


hard_drive_letter="D"
writing_path = "C:/Users/U000000/Documents/apuntamentos-non-opo/TFM/piton"
output_filename_prefix="persim_estable"

pixel_size_dimcero=5#tamaño del pixel de las imágenes de persistencia de dimensión 0
pixel_size_dimuno=4#tamaño del pixel de las imágenes de persistencia de dimensión 1






#######  3- Calculamos las coordenadas de la región de estudio y las subregiones de interés para descartar componentes conexas 


path=hard_drive_letter+":/"+str(an)+"/ARTMIP_MERRA_2D_"+str(an)+me+di+"_"+hor+".nc"
print(path)

ncfile = readnc(path, "r") #leemos los datos


lat=ncfile.variables["lat"][:] #vector de latitudes
la=(lat>=lat_min)&(lat<=lat_max) #nos quedamos con las latitudes que nos interesan (las de la región del estudio)
    

lon=ncfile.variables["lon"][:]#vector de longitudes

lo=(lon>=lon_min)&(lon<=lon_max) #nos quedamos con las longitudes que nos interesan (las de la región del estudio)
  
lats_used=ma.getdata(lat[la]) #filtamos latitudes
lons_used=ma.getdata(lon[lo]) #filtramos longitudes
lathawai=(lat[la]>=lat_min)&(lat[la]<pivot3) #latitudes de hawwai

hawai_matrix = np.zeros((len(lats_used), len(lons_used)), dtype=bool)
    # Usar indexación booleana para asignar True a las filas correspondientes
hawai_matrix[lathawai, :] = True

# Obtener la matrix
america = obtain_boolean_matrix(lats_used, lons_used, pivot1, pivot2, pivot3)
print(america)




###### 4- Definimos funciones


def myfunc(a, b): 
    return a in b
invect=np.vectorize(myfunc) #funcion que nos da una matriz booleana donde TRUE implica que el elemento a de la matriz imput está en el conjunto b


####### 5- Definimos variables de tamaño y vectores para ciertos grupos. Inicializamos listas y valores.

#print(type(tmq))
longlat=lats_used.shape[0] #número de latitudes y de longitudes
longlon=lons_used.shape[0]
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
    hoursvalue=np.array(['00'])
else:
    hoursvalue=np.array(['00', '03', '06', '09', '12', '15', '18', '21'])
diccionario={} ##diccionario que para cada año nos va a decir que número de imagen se corresponde con la primera imagen del año


if proof==1: 
    yearsvalue=np.array([1980])
else:
    yearsvalue=np.arange(1980,2018,1)

nfila=0 ##contador del número de imágenes ya computadas



########6- Bucle de computación de persistencia. Leemos la matriz de datos para cada año, mes, día y hora. Para cada año y para cada mes el número de días del mes varía.
##Primero modificamos la filterfun para que los conjuntos de nivel superior no contengan componentes conexas que no intersequen ni la cossta americana ni las latitudes de hawai
# # y luego calculamos el complejo cúbico y sus persitencias 0 y 1 dimensionales

for year in yearsvalue:



    im=0
    diccionario[year]=nfila
    print(diccionario[year])

    if year==2017:
        monthvalues=np.array([ '01', '02', '03', '04', '05', '06'])
    else:
        monthvalues=np.array([ '01', '02', '03', '04', '05', '06',  '07', '08', '09', '10', '11', '12'])

    if proof==1:
        monthvalues=np.array([ '01', '02'])

    for month in monthvalues:
        print(f"We begin to compute connected components of images of {month}/{year}")

        if month in {"01", "03", "05", "07", "08", "10", "12"}: #meses con 31 días
            daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29','30', '31'])
        if month in {"04", "06", "09","11"}: #meses con 30 días
            daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29','30'])
        if (month=="02")&(year in {1980,1984,1988,1992,1996,2000, 2004,2008,2012,2016}): #año bisiesto febrero tiene 29 días
            daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29'])
            print(year, "was a leap year")        
        if (month=="02")&(year not in {1980,1984,1988,1992,1996,2000, 2004,2008,2012,2016}): #año no bisiesto febrero tiene 28 días
            daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28'])
        
        for day in daysvalues:
            if day in ("01", "10", "20", "28", "29", "30", "31"):
                print("día=",day)
            for hour in hoursvalue:
                if (month== '01' and day in ("01", "10", "20", "28", "29", "30", "31")):
                    print(hour)
                path=hard_drive_letter+":/"+str(year)+"/ARTMIP_MERRA_2D_"+str(year)+month+day+"_"+hour+".nc"
                ncfile = readnc(path, "r") #leemos los datos de esa imagen
                iwp=ncfile.variables["IWV"][la,lo].filled(np.nan) ##cargamos la matriz de IWV en la región del estudio. Si hay valores enmascarados (no hay en nuestro dataset) se pasa a Nan
                im=im+1
                nfila=nfila+1
                #primero modificamos la filterfun para quitarnos ccs que no intersequen las regiones de interés (latitudes de hawai y continente americano)

                filterfun=-iwp
                conxuntocero={0}
                indicador=np.zeros((iwp.shape[0], iwp.shape[1]))

                for l in np.arange(60,- 1, -p): #bucle por umbral donde vamos a calcular las ccs del superlevelset asociado al umbral
                    umbral=-l
                    superlevel=np.uint8(np.array(iwp>l, dtype=np.int16))
                    ccs=cv2.connectedComponents(superlevel)[1]
                    ccshawai=ccs[hawai_matrix]  #cojemos las ccs que intersecan con las latitudes de hawai  
                    if bug==1:
                        ccsinteres=set(ccshawai.tolist()) #con el bug detectado se cuentan solo las ccc q intersequen hawwai
                    else: 
                        ccsinteres=set(ccshawai.tolist())|set(ccs[america].tolist())

                    ccsinteres= ccsinteres  - conxuntocero

                    
                    
                    boolean=invect(ccs, ccsinteres) #matriz que vale TRUE si ese i,j pertenece a una cc

                    indicador=np.where(boolean, indicador+1, indicador) #esta matriz tiene un valor 0 si en ningún momento boolean fue true hasta ahora
                    filterfun= np.where(indicador==1, umbral, filterfun)
                    #cambiamos la filterfun para que no entren ccs en los superlevelsets que no intersequen las regiones de interés (america, hawai) en este umbral 
                    #de esta manera reducimos algún valor de la función en la rejilla hasta que no esté en una cc en los superlevelsets que no intersequen las regiones de interés (america, hawai)
                    #cambiamos signo xq gudhi solo considera sublevel y no superlevel sets
                                    
                #Con este contador medimos cuanto cambia en media y máximo los cambios en la filterfun respecto de los valores iwp originales. Queda comentado pero se puede descomentar:
                contadorcambio=contadorcambio+np.amax(iwp+filterfun)
                contadorcambiomedia=contadorcambiomedia+np.mean(iwp+filterfun) #sumamos a los contadores la media y maximo de los cambios



                filterfun_list =  filterfun.ravel()
                #función de filtrado (lista con el IWV (modificado en el anterior bucle)  siguiendo orden lexicográfico de coordenadas)
                assert iwp.shape == (longlat, longlon) #sale error si no se cumple la cond
                
                cc_river = gd.CubicalComplex( dimensions = [longlat ,longlon], 
                        top_dimensional_cells = filterfun_list
                        ) #construcción del complejo cúbico
                    
                

                persi=cc_river.persistence()    #guardamos la peristencia     
                filtro0 = filter(lambda featur: featur[0] == 0, persi)
                persi0 = [featur[1] for featur in filtro0 if np.isfinite(featur[1][1])]   #lista con los intervalos de dim 0 y su dim (0 siempre)  
                #eliminamos el intervalo infinito

                persis0.append(persi0)
                #edad_muerte_0=list(map(tiempovida, persi0)) #lista con la longitud de intervalos de persistencia de dim 0 de la imagen (menos el primero)
                #edades_muerte_0.append(edad_muerte_0)


                filtro1 = filter(lambda featur: featur[0] == 1, persi)
                persi1=list(filtro1) #lista con los intervalos de dim 0 y su dim (1 siempre)
                persi1=[featur[1] for featur in persi1] #lista con los intervalos de persistencia de dim 1 de la imagen 
                persis1.append(persi1)
                #edad_muerte_1=list(map(tiempovida, persi1)) #lista con la longitud de intervalos de persistencia de dim 0 de la imagen
                #edades_muerte_1.append(edad_muerte_1)

                ### Limpiamos la memoria. Estas variables son internas del bucle
                del iwp, filterfun, boolean, indicador, cc_river, persi
                if im % 100 == 0:
                    gc.collect()
    
    print("en ano", year, "hubo", im, "imágenes")
print("hay un total de", nfila, "imágenes")



#######7- CALCULAMOS LAS IMÁGENES DE PERSISTENCIA DE DIM=1



pimgr = PersistenceImager(pixel_size=pixel_size_dimuno) #definimos la clase imagen de persistencia y ponemos una precisión

pdgms = [np.asarray(persis1[im], dtype=float) for im in range(nfila)] #lista donde para cada imagen obtenemos un array con los intervalos de persistencias (los intervalos de persistencias son listas y no tuples)



pimgr.fit(pdgms, skew=True) #ajustamos el tamaño de imagen al rango máximo en la lista de las persistencias
print(pimgr)
tamaño_persis=pimgr.resolution ##vemos el tamaño de la imagen de persistencia

print(tamaño_persis)

del persis1 #borramos datos no necesarios
gc.collect()

pimgs = pimgr.transform(pdgms, skew=True, n_jobs=-1) #lista donde tenemos para cada im un array con los valores de la densidad en cada rejilla
data = [] #lista donde para cada imagen añadiremos un vector con año, imagen, persimdim1 y persimdim0 aplanadas

# Precalcular límites de año (solo una vez)
sorted_years = sorted(diccionario.items(), key=lambda x: x[1])  # [(año, índice inicial), ...]
year_indices = [v for _, v in sorted_years]
years_sorted = [k for k, _ in sorted_years]


for im in np.arange(0, nfila, 1):  ##para cada im añadimos en la lista data el año correspondiente, el número de la imagen en el año y la imagen de persistencia de dim 1 aplanada 
    xval = np.arange(0,tamaño_persis[0] , 1)
    yval = np.arange(0,tamaño_persis[1] , 1)
    valores_dicc=list(diccionario.values()) #calculamos el año en el que está la imagen a través de diccionario (tiene como llaves los años y el índice de la primera imagen )
    # Buscamos el último año cuyo índice inicial ≤ im
    idx = np.searchsorted(year_indices, im, side="right") - 1
    ano = years_sorted[idx]
    imaxe_do_ano = im - diccionario[ano]
    if daily==1:
        data.append([ano,imaxe_do_ano*8]+pimgs[im].ravel().tolist()) #añadimos el array como lista aplanada a data
    else: 
        data.append([ano,imaxe_do_ano]+pimgs[im].ravel().tolist()) 

    

pdgms=[]#borro la memoria ya que vamos a reiniciar con dim 1
pimgs=[]


#######8- CALCULAMOS LAS IMÁGENES DE PERSISTENCIA DE DIM=0
pimgr = PersistenceImager(pixel_size=pixel_size_dimcero) #definimos la clase imagen de persistencia y ponemos una precisión
pdgms = [np.asarray(persis0[im], dtype=float) for im in range(nfila)]  #lista donde para cada imagen obtenemos un array con los intervalos de persistencias (los intervalos de persistencias son listas y no tuples)

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
    data[im].extend(pimgs[im].ravel().tolist()) #añadimos el array como lista aplanada a data[im] sumandola a la anterior

del pdgms #borramos datos no necesarios
del pimgs
gc.collect()

####### 9- Guardamos resultados
x=pd.DataFrame(data) #hacemos data frame la lista data (de la vectorización de las persims)
print("dataframe with persistence images done")

if (proof==0 and p==0.25):
    if daily==1:
        x.to_csv(f"{writing_path}/{output_filename_prefix}_daily{pixel_size_dimuno}{pixel_size_dimcero}_{pivot2}{bug_text}_temp.csv")
    else:
        x.to_csv(f"{writing_path}/{output_filename_prefix}{pixel_size_dimuno}{pixel_size_dimcero}_{pivot2}{bug_text}_temp.csv")
    print("saved as csv")



del x
gc.collect()

