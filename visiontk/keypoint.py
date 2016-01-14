# -*- coding: utf-8 -*-

from collections import namedtuple
import pprint

import numpy as np

class Keypoint(object):
   def __init__(self, position, scale, orientation, descriptor):
      self.position = position
      self.scale = scale
      self.orientation = orientation
      self.descriptor = descriptor
      
   def __str__(self):
      s = "(%f, %f), scale=%f, orient=%f\n" % (self.position[0], self.position[1], self.scale, self.orientation)
      s2 = "%s\n" % pprint.pformat(self.descriptor, width=128*4)
      return ''.join([s, s2])
      
   def __eq__(self, other):
      return self.__dict__ == other.__dict__


class SiftKeypoint(Keypoint):
   # Origin (0,0) is at the top-left of the image.
   # Axes : (row, column) or (x,y) 
   #    +--> y
   #    |
   #  x v
   def __init__(self, position, scale, orientation, descriptor):
      super(SiftKeypoint, self).__init__(position, scale, orientation, descriptor)
      
   def __str__(self):
      s = "(%f, %f), scale=%f, orient=%f\n" % (self.position[0], self.position[1], self.scale, self.orientation)
      s2 = "%s\n" % pprint.pformat(self.descriptor, width=128*4)
      return ''.join([s, s2])
      
   def __eq__(self, other):
      return self.__dict__ == other.__dict__

OpenCVkeypoint = namedtuple('OpenCVkeypoint', ('pt','size','angle','response','octave','class_id'))

def SIFT2OpenCV(keypoint):
   x = keypoint.position[0]
   y = keypoint.position[1]

   kp = OpenCVkeypoint(
      pt = np.array([y,x]), # Swap coordinates to respect OpenCV camera model
      size = None,
      angle = keypoint.orientation,
      response = None,
      octave = keypoint.scale,
      class_id = None)

   return {'keypoint' : kp,
           'descriptor': keypoint.descriptor}

def convert_SIFT2OpenCV(keypoints):
   cvKeypoints = []
   cvDescriptors = []

   for kp in keypoints:
      cvKp = SIFT2OpenCV(kp)
      cvKeypoints.append(cvKp['keypoint'])
      cvDescriptors.append(cvKp['descriptor'])

   return {'keypoints' : cvKeypoints,
           'descriptors': cvDescriptors}
