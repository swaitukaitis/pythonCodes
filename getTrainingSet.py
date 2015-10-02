# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 2015
@author: swaitukaitis
"""
import Image
import os as os
import numpy as np
import pickle

def getTrainingSet(folder):
    """
    Reads in images and outputs them as training data.  The images should live in 
    "folder/happy/*" and "folder/frowny/*"  The program converts each to grayscale, resizes
    it to a thumbnails (dimensions given by "size"), and unrolls it into a vector.  Finally,
    we return variables X & y (all training data), Xval and yval (all cross validation data).    
    """
    #size of thumbnails
    size=20, 20 
    
    #first get the happy faces
    happyfiles=[ f for f in os.listdir(folder+'happy') if f.endswith(".jpg") ]
    X=np.zeros([np.size(happyfiles), size[0]**2])
    y=np.repeat(1, np.size(happyfiles))
    print "Happy Faces!"
    for i in np.arange(np.size(happyfiles)):
        im = Image.open(folder+'happy/'+happyfiles[i]).convert('LA')
        box=[int(np.floor(im.size[0]/2.-(np.min(im.size)-1)/2.)),int(np.floor(im.size[1]/2.-(np.min(im.size)-1)/2.)),int(np.floor(im.size[0]/2.-(np.min(im.size)-1)/2.))+np.min(im.size)-1, np.min(im.size)-1+int(np.floor(im.size[1]/2.-(np.min(im.size)-1)/2.))]
        im = im.crop(box=box)
        im.thumbnail(size)
        vals=np.asarray(im.convert('L')).reshape(size[0]**2)
        if vals.max > 0:
            X[i,:] = np.asarray(im.convert('L')).reshape(size[0]**2)
        print np.float(i)/np.size(happyfiles), "\r",
    
    #Get the frowny faces
    frownyfiles=[ f for f in os.listdir(folder+'frowny') if f.endswith(".jpg") ]
    X=np.vstack([X, np.zeros([np.size(frownyfiles), size[0]**2])])
    y=np.hstack([y, np.zeros(np.size(frownyfiles))])
    print "Frowny Faces!"
    for i in np.arange(np.size(frownyfiles)):
        im = Image.open(folder+'frowny/'+frownyfiles[i]).convert('LA')
        box=[int(np.floor(im.size[0]/2.-(np.min(im.size)-1)/2.)),int(np.floor(im.size[1]/2.-(np.min(im.size)-1)/2.)),int(np.floor(im.size[0]/2.-(np.min(im.size)-1)/2.))+np.min(im.size)-1, np.min(im.size)-1+int(np.floor(im.size[1]/2.-(np.min(im.size)-1)/2.))]
        im = im.crop(box=box)
        im.thumbnail(size)
        X[i+np.size(happyfiles),:] = np.asarray(im.convert('L')).reshape(size[0]**2)
        print np.float(i)/np.size(frownyfiles), "\r",
    
    #Separate the training set from the validation set
    valFrac=0.2
    valInd=np.random.choice(X.shape[0], np.round(X.shape[0]*valFrac))
    trainInd=np.setdiff1d(np.arange(X.shape[0]), valInd)
    
    #Save a copy of the data to the folder
    pickle.dump([X, y], open(folder+'data.p', 'wb'))
    
    #And return this data to the user
    return(X, y)
