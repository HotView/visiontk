# -*- coding: utf-8 -*-

import gzip
import visiontk

class KeyFile(object):
   def __init__(self, filename):
      self.filename = filename
      self.keypoints = []
      
      self.parse()
      
   def parse(self):
      self.keypoints = read_SIFT(self.filename)
      
   def getKeypoint(self, index):
      return self.keypoints[index]


def read_SIFT(filename):
   f = gzip.open(filename, "r")
   keypoints = []
      
   try:
      header = f.readline()
      nFeatures, vectorLength = map(lambda x : int(x), header.split())
      
      for i in range(nFeatures):
         line = f.readline()

         # Origin (0,0) is at the top-left of the image.
         # Axes : (row, column) or (x,y) 
         #    +--> y
         #    |
         #  x v
         row, col, scale, orientation = map(lambda x : float(x), line.split())
         
         descriptor = []
         while len(descriptor) < vectorLength:
            line = f.readline()
            descriptor.extend(map(lambda x : int(x), line.split()))
            
         assert len(descriptor) == vectorLength
         
         keypoints.append(visiontk.keypoint.Keypoint((row,col), scale, orientation, descriptor))
   finally:
      f.close()
   return keypoints