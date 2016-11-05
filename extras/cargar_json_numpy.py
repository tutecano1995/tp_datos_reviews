# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 17:30:44 2016

@author: joaquintz
"""
import json
import numpy as np
import base64
in_file="/home/joaquintz/Desktop/facultad/datos/boc/tsne_10k_rand_cos.json"
label_file="/home/joaquintz/Desktop/facultad/datos/boc/tsne_10k_rand_cos_estrellas.json"

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


print "Loading from "+in_file
with open(in_file, 'r') as f:
    reduced_set = json.load(f,object_hook=json_numpy_obj_hook)

with open(label_file,'r') as f:
    labels = json.load(f,object_hook=json_numpy_obj_hook)
 
