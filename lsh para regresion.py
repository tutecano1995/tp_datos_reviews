# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import csv
import json

import math

def shingles(longitud,text):
    shingles=[]
    for i in range(len(text)-longitud+1):
        shingles.append(text[i:i+longitud])

    return shingles

def hash_u(x,a,b):
	return ((a*hash(x)+b) % 19999999769)

def minhash(shingles):
    h=[199999999999999  for x in range(cantidad_hashes)]
    for shingle in shingles:
        for i in range(cantidad_hashes):
     
            h[i]=min(h[i], hash_u(shingle,i+1,1037))
            
    return h
def lista_a_tupla(lista):
    cadena=""    
    for numero in lista:
        cadena+=str(numero)+" "
    return cadena
cero=0
unoymedio=0
unobis=0	
uno=0
dos=0
tres=0
cuatro=0
cinco=0
seis=0

def prediccion(texto,resumen,longitud_shingles,longitud_resumenes,productId,clientId):
	##promedio
    global cero
    global uno
    global unoymedio
    global unobis
    global dos
    global tres
    global cuatro
    global cinco
    global seis    
    cero+=1
    minhashes=minhash(shingles(longitud_shingles,texto))
    cantidad=0
    suma=0
    for i in range(b):
        tupla=lista_a_tupla(minhashes[i*(cantidad_hashes/b):(i+1)*(cantidad_hashes/b)])
        if tupla in tablas[i]:
            for puntaje  in tablas[i][tupla]:
                suma+=puntaje
                cantidad+=1
    if(cantidad!=0):
        return suma/float(cantidad)
    uno+=1
    cantidad=0
    suma=0
#    for i in range(bmaspermisivo):
#        tupla=lista_a_tupla(minhashes[i*(cantidad_hashes/bmaspermisivo):(i+1)*(cantidad_hashes/bmaspermisivo)])
#        if tupla in tablasmaspermisivo[i]:
#            for puntaje  in tablasmaspermisivo[i][tupla]:
#                suma+=puntaje
#                cantidad+=1
#    if(cantidad!=0):
#        return suma/float(cantidad)
    unoymedio+=1
    cantidad=0
    suma=0
    minhashesresumen=minhash(shingles(longitud_resumenes,resumen))
    for i in range(bresumenmenospermisivo):
        tupla=lista_a_tupla(minhashesresumen[i*(cantidad_hashes/bresumenmenospermisivo):(i+1)*(cantidad_hashes/bresumenmenospermisivo)])
        if tupla in tablasresumenmenospermisivo[i]:
            for puntaje  in tablasresumenmenospermisivo[i][tupla]:
                suma+=puntaje
                cantidad+=1
    if(cantidad!=0):
        return suma/float(cantidad)
    unobis+=1
    cantidad=0
    suma=0
    for i in range(bresumen):
        tupla=lista_a_tupla(minhashesresumen[i*(cantidad_hashes/bresumen):(i+1)*(cantidad_hashes/bresumen)])
        if tupla in tablasresumen[i]:
            for puntaje  in tablasresumen[i][tupla]:
                suma+=puntaje
                cantidad+=1
    if(cantidad!=0):
        
        return suma/float(cantidad)    
    dos+=1
    if (clientId in clientes and clientes[clientId][2]<0.5):
        return clientes[clientId][1]
    tres+=1
    if (productId in productos and productos[productId][2]<0.5):
        return productos[productId][1]
    cuatro+=1
#    if(clientId in clientes):
#        return clientes[clientId][1]
#    cinco+=1
#    if(productId in productos):
#        return productos[productId][1]
    seis+=1
    return 4.1
#
#with open('resultados3.csv','w') as resultado_file:
#    resultado_csv=csv.writer(resultado_file)
#    resultados={}
#    
#    bes=[2,3,6,12,24]
#    longitudes=[35, 50]
#    for b in bes:
#        for longitud in longitudes:

cuatro=0
cinco=0
seis=0

cantidad_hashes=48
   
b=24
longitud=35

bresumen=16
longitudresumen=3

bresumenmenospermisivo=6

bmaspermisivo=48
clients={}
products={}
print("start")
tablas=[{} for j in  range(b) ]

tablasmaspermisivo=[{} for j in  range(bmaspermisivo) ]
tablasresumen=[{} for j in range(bresumen)]
tablasresumenmenospermisivo=[{} for j in range(bresumenmenospermisivo)]
print("termino crear tabla")
with open('/home/gonza/Datos/BOC/train_processed.csv','r') as train_file:
    train_csv = csv.reader(train_file)
    next(train_csv)
    j=0
    for row in train_csv:
        posiciones=minhash(shingles(longitud,row[10]))
        posicionesresumen=minhash(shingles(longitudresumen,row[9]))
        
        puntaje=int(row[7])
        for i in range(bmaspermisivo):
            tupla=lista_a_tupla(posiciones[i*(cantidad_hashes/bmaspermisivo):(i+1)*(cantidad_hashes/bmaspermisivo)])
                        
            if(not tupla in tablasmaspermisivo[i]):
                tablasmaspermisivo[i][tupla]=[]
            tablasmaspermisivo[i][tupla].append(puntaje)
        for i in range(b):
            tupla=lista_a_tupla(posiciones[i*(cantidad_hashes/b):(i+1)*(cantidad_hashes/b)])
                        
            if(not tupla in tablas[i]):
                tablas[i][tupla]=[]
            tablas[i][tupla].append(puntaje)
        for i in range(bresumen):
            tupla=lista_a_tupla(posicionesresumen[i*(cantidad_hashes/bresumen):(i+1)*(cantidad_hashes/bresumen)])
            if(not tupla in tablasresumen[i]):
                tablasresumen[i][tupla]=[]
            tablasresumen[i][tupla].append(puntaje)
        for i in range(bresumenmenospermisivo):
            tupla=lista_a_tupla(posicionesresumen[i*(cantidad_hashes/bresumenmenospermisivo):(i+1)*(cantidad_hashes/bresumenmenospermisivo)])
            if(not tupla in tablasresumenmenospermisivo[i]):
                tablasresumenmenospermisivo[i][tupla]=[]
            tablasresumenmenospermisivo[i][tupla].append(puntaje)
            
clientes={}
productos={}
with open('/home/gonza/Datos/BOC/train_processed.csv','r') as train_file:
    train_csv = csv.reader(train_file)
    next(train_csv)
    j=0
    for row in train_csv:
        if not row[3] in clientes:
            clientes[row[3]]=[]
        clientes[row[3]].append(int(row[7]))
        if not row[2] in productos:
            productos[row[2]]=[]
        productos[row[2]].append(int(row[7]))
        
        
for clientId, puntajes in clientes.iteritems():
    suma=0
    for puntaje in puntajes:
        suma+=puntaje
    promedio=suma/float(len(puntajes))
    suma=0    
    for puntaje in puntajes:
        suma+=(puntaje-promedio)**2
    if(len(puntajes)!=1):
        varianza=suma/float(len(puntajes)-1) #estimador insesgado
    else:
        varianza=9999
    clientes[clientId]=(len(puntajes),promedio,varianza)

for productId, puntajes in productos.iteritems():
    suma=0
    for puntaje in puntajes:
        suma+=puntaje
    promedio=suma/float(len(puntajes))
    suma=0    
    for puntaje in puntajes:
        suma+=(puntaje-promedio)**2
    if(len(puntajes)!=1):
        varianza=suma/float(len(puntajes)-1)  #estimador insesgado
    else:
        varianza=9999
    productos[productId]=(len(puntajes),promedio,varianza)

    
#with open('lsh.csv','w') as archivo:
	#json.dump(tablas,archivo)
 
#with open('/home/gonza/Datos/BOC/validation.csv','r') as validation_file:
#    validation_csv=csv.reader(validation_file)
#    j=0
#    EMS=0
#    for row in validation_csv:
#        pred=prediccion(row[10],longitud)
#        #print pred
#        EMS+=(int(row[7])-pred)**2
#        j=j+1
#    EMS/=float(j)
#print EMS
#print b
#print longitud
#resultados[(b,longitud)]=EMS
#resultado_csv.writerow([b,longitud,EMS])
print cero
print uno
print unoymedio
print unobis
print dos
print tres
print cuatro
print cinco
print seis


j=0
with open('/home/gonza/Datos/BOC/data/test_processed.csv','r') as test_file:
    test_csv=csv.reader(test_file)
    with open('/home/gonza/Datos/BOC/submission.csv','w') as submission_file:
        submission=csv.writer(submission_file)
        next(test_csv)
        submission.writerow(["Id","Prediction"])
        
        for row in test_csv:
            submission.writerow([row[1],prediccion(row[9],row[8],longitud,longitudresumen,row[2],row[3])])
            if not j%10000:
			print j
            j=j+1
            
