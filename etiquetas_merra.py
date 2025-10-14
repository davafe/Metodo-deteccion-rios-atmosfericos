
#######   1- Carga de librerías

from netCDF4 import Dataset as readnc
#import gudhi as gd  
import numpy as np
import pandas as pd
#import cv2
import numpy.ma as ma
#import matplotlib.pyplot as plt


#import rioxarray as rx

####### 1.1 - Importing local modules

from local_functions.matrix import *

#######   2- Parametros y definición data frame


x=pd.DataFrame(columns=["ano", "imagen", "AR"]) #creamos un data frame dónde incluiremos para cada imagen el año, núm de imagen (dentro de ese año) y la etiqueta de si es AR


writing_path = "C:/Users/U000000/Documents/apuntamentos-non-opo/TFM/piton"
output_filename_prefix="etiquetas_merra_2"
hard_drive_letter="E"
carpeta_merra_labels="merra_labels"
direccion=f"{hard_drive_letter}:/{carpeta_merra_labels}/MERRA2.ar_tag.cascade_bard_v1.3hourly."+str(1980)+".nc4"
#direccion="C:/Users/David/OneDrive/Documents/mestrado/tfm-solicitude bolsa/piton/MERRA2.ar_tag.cascade_bard_v1.3hourly."+str(1980)+".nc4"
daily=0 #daily=1 si usamos solo pricisión temporal de 24 h. daily=!1-> precisión temporal de 3 h 
pivote1=47.8
pivote2=40.25
pivote3=22.8
lat_min=19
lat_max=56
lon_min=-179
lon_max=-109


#####  3- Calculamos las coordenadas de la región de estudio y las subregiones de interés para descartar componentes conexas 

ncfile = readnc(direccion, "r") #cargamos primero los datos para obtener (para todos los años) lon índices de lat y long que vamos a usar
#tb sacamos lon indices que se corresponden con las lat_hawai y con el continente americano
lat=ncfile.variables["lat"][:]
print(lat)
type(lat)

la=(lat>=lat_min)&(lat<=lat_max) #true las latitudes de la región que estudiamos
print(lat[la], la.shape, lat[la].shape)


lon=ncfile.variables["lon"][:]
print(lon)
type(lon)

lo=(lon>=lon_min)&(lon<=lon_max)  #true las longitudes de la región que estudiamos
print(lon[lo], lo.shape, lon[lo].shape)



lats_usadas=ma.getdata(lat[la])
lons_usadas=ma.getdata(lon[lo])
num_lats_usadas=lats_usadas.shape[0]
num_lons_usadas=lons_usadas.shape[0]


                                                                    
lathawai=(lat[la]>=19)&(lat[la]<22.8) #vector dónde hay un TRUE si ese índice se corresponde con la latitud de HAwai


region_america=obtain_boolean_matrix(lats_usadas, lons_usadas, pivote1, pivote2, pivote3)
                                                                        
region_hawai = np.tile(lathawai[:, np.newaxis], (1, num_lons_usadas))


                                                                        








########### 6- Bucle cálculo de etiquetas y las cargamos en x
for year in np.arange(1980, 2018, 1): #para cada año cargamos los datos y sacamos las etiquetas de AR para cada imagen
    
    direccion=f"{hard_drive_letter}:/{carpeta_merra_labels}//MERRA2.ar_tag.cascade_bard_v1.3hourly."+str(year)+".nc4"
    #direccion="C:/Users/David/OneDrive/Documents/mestrado/tfm-solicitude bolsa/piton/MERRA2.ar_tag.cascade_bard_v1.3hourly."+str(year)+".nc4"
    ncfile = readnc(direccion, "r")

    tmqlongtemp=ncfile.variables["ar_binary_tag"].shape[0] #cargamos el número de imágenes de ese año (con precisión 3-hourly)
    
    #cargamos el array de etiquetado geográfico de ARs (solo en región de estudio) (lo hacemos daily o no dependiendo del valor de daily)

    if daily==1:
        etiquetas=ma.getdata(ncfile.variables["ar_binary_tag"][np.arange(0,tmqlongtemp,8),la,lo])
    else:
        etiquetas=ma.getdata(ncfile.variables["ar_binary_tag"][:,la,lo])

    for im in np.arange(0, etiquetas.shape[0], 1): #para cada imagen sacamos etiqueta AR=1 sii hay algún 1 de AR en lat-hawai y hay algún 1 de AR en américa
        etiq=etiquetas[im,:,:].astype(bool)
        y=0
        if (np.any(np.logical_and(region_hawai, etiq)) and np.any(np.logical_and(region_america, etiq))):
            y=1
        if daily==1:
            x.loc[len(x),:]=[year, im*8, y] 
        else:
            x.loc[len(x),:]=[year, im, y] 

    filter=x['ano']==year
    suma=x[filter]["AR"].sum() #número de etiquetas AR=1 en este año
    print(year, suma, suma/etiquetas.shape[0])

########### 7-cargamos datos a un csv
if daily==1:
    #x.to_csv(f"{writing_path}/etiquetas_merra_daily.csv")
    #x.to_csv(f"{writing_path}/pruebatonta_daily.csv")
    x.to_csv(f"{writing_path}/{output_filename_prefix}_daily.csv")

else:
    #x.to_csv(f"{writing_path}/etiquetas_merra.csv")
    #x.to_csv(f"{writing_path}/pruebatonta.csv")
    x.to_csv(f"{writing_path}/{output_filename_prefix}.csv")
