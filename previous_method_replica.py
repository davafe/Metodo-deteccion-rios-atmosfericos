#######   1- Carga de librerías

#from email.headerregistry import ContentTypeHeader # no se usa
from netCDF4 import Dataset as readnc
import numpy as np
import pandas as pd
import math
import numpy.ma as ma
import scipy.misc 
import itertools as it
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
daily = 0 #vale 1 si consideramos solo datos con precisión diaria y no 3-horaria (cojemos los datos de las imágenes 0,8,16, ... corespondientes a las 00)
proof = 0 # vale 1 si hacemos prueba con muchos menos imágenes

hard_drive_letter="D"
writing_path = "C:/Users/U000000/Documents/apuntamentos-non-opo/TFM/piton"
output_filename_prefix="cv2_vectors"

p= 1 
thresholds = np.arange(-60, 1, p )



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

####### 4- Bucle de cálculo

if daily==1:
    horasvalue=np.array(['00'])
else:
    horasvalue=np.array(['00', '03', '06', '09', '12', '15', '18', '21'])

if proof==1: 
    yearsvalue=np.array([1980])
else:
    yearsvalue=np.arange(1980,2018,1)

diccionario={} ##diccionario que para cada año nos va a decir que número de imagen se corresponde con la primera imagen del año


nfila=0 ##contador del número de imágenes ya computadas


if daily==1:
    hoursvalue=np.array(['00'])
else:
    hoursvalue=np.array(['00', '03', '06', '09', '12', '15', '18', '21'])
vectors=[]
image_number_in_year=0
for year in yearsvalue:

    image_number_in_year=0
    diccionario[year]=nfila
    print(diccionario[year])

    if year==2017:
        monthvalues=np.array([ '01', '02', '03', '04', '05', '06'])
    else:
        monthvalues=np.array([ '01', '02', '03', '04', '05', '06',  '07', '08', '09', '10', '11', '12'])

    if proof==1:
        monthvalues=np.array([ '01', '02'])

    for month in monthvalues:
            print("We begin to compute connected components of images of "+ month+ "/", year )

            if month in {"01", "03", "05", "07", "08", "10", "12"}: #meses con 31 días
                daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29','30', '31'])
            if month in {"04", "06", "09","11"}: #meses con 30 días
                daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29','30'])
            if (month=="02")&(year in {1980,1984,1988,1992,1996,2000, 2004,2008,2012,2016}): #año bisiesto febrero tiene 29 días
                daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29'])
                print(month, "foi bisiesto")        
            if (month=="02")&(year not in {1980,1984,1988,1992,1996,2000, 2004,2008,2012,2016}): #año no bisiesto febrero tiene 28 días
                daysvalues=np.array(['01', '02', '03','04', '05', '06','07', '08', '09',"10", '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28'])

            for day in daysvalues:
                print("día=",day)
                for hour in hoursvalue:
                    if month== '01':
                        print(hour)
                    path=hard_drive_letter+":/"+str(year)+"/ARTMIP_MERRA_2D_"+str(year)+month+day+"_"+hour+".nc"
                    ncfile = readnc(path, "r") #leemos los datos de esa imagen
                    iwp=ncfile.variables["IWV"][la,lo] ##cargamos la matriz de IWV en la región del estudio

                    # Ejemplo de cómo llamar a la función:
                    minus_iwp = -iwp  # Aplicar el negativo de la matriz iwp
                    component_size_2 = calculate_component_sizes_with_elder(minus_iwp, thresholds, hawai_matrix, america)

                    # Reverse the component_size array efficiently
                    component_sizes_2 = component_size_2[::-1]

                    #We add the result vector to the list
                    if daily==1:
                        vectors.append([year, image_number_in_year*8]+component_sizes_2)
                    else: 
                        vectors.append([year, image_number_in_year]+component_sizes_2)

                    image_number_in_year=image_number_in_year+1 #we actualize the counting of images
                    # Result
                    #print("Component size vector:", component_sizes_2)

output=pd.DataFrame(vectors)

if daily==1:
    output.to_csv(f"{writing_path}/{output_filename_prefix}_{p}_daily_until_{month}_{year}_{pivot2}.csv")
else:  
    output.to_csv(f"{writing_path}/{output_filename_prefix}_{p}_until_{month}_{year}_{pivot2}.csv")

