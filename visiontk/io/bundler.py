# -*- coding: utf-8 -*-

import numpy as np
import pprint
import copy
from collections import Iterable

import visiontk as vtk
import visiontk.keypoint as sift

class BundlerCamera(vtk.Camera):
   def __init__(self, f, k1, k2, R, t):
      # f : focal length
      # k1, k2 : radial distortion coefficients
      # R : 3x3 rotation matrix
      # t : 1x3 translation vector

      disto_coeffs = vtk.camera.DistorsionCoeffients((k1,k2),(0,))
      c = (0,0)

      intrinsics = vtk.Intrinsics(f=f, c=c, disto_coeffs=disto_coeffs)
      extrinsics = vtk.Extrinsics(R=R, t=t)
      super(BundlerCamera, self).__init__(intrinsics=intrinsics, extrinsics=extrinsics)

   @property
   def f(self):
       return self.intrinsics.f[0]

   @property
   def k1(self):
       return self.intrinsics.disto_coeffs.radial[0]
      
   @property
   def k2(self):
       return self.intrinsics.disto_coeffs.radial[1]

   @property
   def R(self):
       return self.extrinsics.R

   @property
   def t(self):
       return self.extrinsics.t

   def to_extended_bundler(self, img_dimensions, K=None):
      intrinsics = copy.copy(self.intrinsics)
      intrinsics.img_dimensions = img_dimensions
      
      if K is not None:
         intrinsics.K = K
      else:
         intrinsics.c = img_dimensions / 2.0

      # 180 degree rotation around x-axis.
      # Bundler camera point in negative z, whereas OpenCV (and Extended Bundler) cameras point toward positive z.
      z_inverse = np.diag([1.0, -1.0, -1.0])
      extrinsics = vtk.Extrinsics(np.dot(z_inverse, self.extrinsics.R), np.dot(z_inverse, self.extrinsics.t))

      return ExtendedBundlerCamera(intrinsics, extrinsics)

   def __str__(self):
      lines = ["f=%f, k1=%f, k2=%f\n" % (self.f, self.k1, self.k2)]
      lines.append("%s\n" % pprint.pformat(self.R))
      lines.append("%s\n" % pprint.pformat(self.t))
      return ''.join(lines)

class ExtendedBundlerCamera(vtk.Camera):
   def __init__(self, intrinsics, extrinsics):
      super(ExtendedBundlerCamera, self).__init__(intrinsics=intrinsics, extrinsics=extrinsics)

      k = list(self.intrinsics.disto_coeffs.radial)
      k.append(0.0)
      self.intrinsics.disto_coeffs.radial = k

   def project(self, point3d):
      pt = np.dot(self.R, point3d) + self.t.flatten()
      pt = pt[0:2] / pt[-1]
      pt = np.dot(np.diag(self.f), pt) + self.principal_point
      return pt

   @classmethod
   def from_BundlerCamera(cls, bundler_cam):
      cam = cls(intrinsics=bundler_cam.intrinsics, extrinsics=bundler_cam.extrinsics)
      return cam

   @property
   def f(self):
       return self.intrinsics.f

   @property
   def disto_coeffs(self):
      return self.intrinsics.disto_coeffs.radial

   @property
   def principal_point(self):
      return self.intrinsics.c

   @property
   def R(self):
       return self.extrinsics.R

   @property
   def t(self):
       return self.extrinsics.t

   def __str__(self):
      lines = ["f=%s, c=%s, k=%s\n" % tuple(map(pprint.pformat,(self.f, self.principal_point, self.disto_coeffs)))]
      lines.append("%s\n" % pprint.pformat(self.R))
      lines.append("%s\n" % pprint.pformat(self.t))
      return ''.join(lines)

class View(object):
   # The coordinate system origin (0,0) is centered on the image center
   # (w/2,h/2). The 'x' axis increases to the right and the 'y' axis increases
   # to the top. So, the bottom-left corner is at (-w/2, -h/2).
   def __init__(self, camera, key, x, y):
      self.camera = camera
      self.key = key
      self._position = np.array([x,y])

   @property
   def position(self):
      return self._position

   @position.setter
   def position(self, value):
      self._position = value.reshape((2,))

   @property
   def x(self):
       return self._position[0]

   @x.setter
   def x(self, value):
       self._position[0] = value

   @property
   def y(self):
       return self._position[1]

   @y.setter
   def y(self, value):
       self._position[1] = value   
      
   def __str__(self):
      return "camera=%i, key=%i, x=%f, y=%f" % (self.camera, self.key, self.x, self.y)

   def __repr__(self):
      return self.__str__()

class Point(object):
   def __init__(self, position, color, viewList):
      self.pos = position # 1x3 vector
      self.color = color  # RGB vector
      self.viewList = viewList
      
      self.views = {}
      self._build_viewmap()
   
   @classmethod
   def copy_with_view_subset(cls, point, cam_IDs):
      view_list = select_views(point.viewList, cam_IDs)
      new_point = cls(point.pos, point.color, view_list)
      return new_point

   def _build_viewmap(self):
      self.views = {}
      for view in self.viewList:
         self.views[view.camera] = view

   def __str__(self):
      lines = ["pos=(%f,%f,%f)\nRGB=(%i,%i,%i)\n" % (self.pos[0], self.pos[1], self.pos[2], self.color[0], self.color[1], self.color[2])]
      for view in self.viewList:
         lines.append("%s" % view)
      return ''.join(lines)
      
   def getCameraView(self, camera):
      try:
         view = self.views[camera]
      except KeyError:
         raise KeyError("No view in camera %i" % camera)
      return view

   def isVisibleInCams(self, camID_list):
      if not isinstance(camID_list, Iterable):
         camID_list = (camID_list,)
      return all([camID in self.views for camID in camID_list])

def select_views(views, cam_IDs):
   selected_views = []
   for v in views:
      if v.camera in cam_IDs:
         selected_views.append(v)
   return selected_views

class BaseBundlerParser(object):
   def __init__(self, filePath):
      self.filePath = filePath
      self.cameraList = []      
      self.pointList = []
      self.constraints = []
      
      self.parse()
      
   def parse(self):
      with open(self.filePath, "r") as f:
         header = f.readline()
         itemCount = f.readline()
         
         objCounts = map(lambda x : int(x), itemCount.split())

         numCams = objCounts[0]
         numPoints = objCounts[1]
         if len(objCounts) > 2:
            numConstraints = objCounts[2]
         else:
            numConstraints = 0

         self.parseConstraints(f, numConstraints)
         self.parseCams(f, numCams)
         self.parsePoints(f, numPoints)

   def parseConstraints(self, file, numConstraints):
      raise NotImplementedError("parseConstraints")

   def parseCams(self, file, numCams):
      raise NotImplementedError("parseCams")
   
   def parsePoints(self, file, numPoints):
      for pointIndex in range(numPoints):
         line = file.readline()
         position = map(lambda x : float(x), line.split())
         
         line = file.readline()
         color = map(lambda x : int(x), line.split())
         
         views = self.parseViewList(file.readline())
         
         self.pointList.append(Point(position, color, views))
         
   def parseViewList(self, line):
      elems = line.split()
      length = int(elems[0])
      viewList = []
      VIEW_LENGTH = 4
      
      for i in range(length):
         base = 1 + i * VIEW_LENGTH
         camera, key, x, y = elems[base : base + 4]
         
         viewList.append(View(int(camera), int(key), float(x), float(y)))
         
      return viewList
      
   def pointsToPixels(self, cameraIndex):
      camera = self.cameraList[cameraIndex]
      
      return [pointToPixel(point, camera) for point in self.pointList], [point.color for point in self.pointList]
      
   def projectPoints(self, cameraIndex):
      camera = self.cameraList[cameraIndex]
      
      return [projectPoint(point, camera) for point in self.pointList], [point.color for point in self.pointList]

   @property
   def tracks(self):
      return map(point2track, self.pointList)

class BundlerOut(BaseBundlerParser):
   def __init__(self, filePath):
      super(BundlerOut, self).__init__(filePath)

   def parseConstraints(self, file, numConstraints):
      assert(numConstraints == 0)

   def parseCams(self, file, numCams):
      for camIndex in range(numCams):
         line = file.readline()
         f, k1, k2 = map(lambda x : float(x), line.split())
         
         # Parse R matrix
         R = []
         for i in range(3):
            line = file.readline()            
            R.append(map(lambda x : float(x), line.split()))
            
         line = file.readline()
         t = map(lambda x : float(x), line.split())
         
         self.cameraList.append(BundlerCamera(f, k1, k2, np.array(R), np.array(t)))

class ExtendedBundlerOut(BaseBundlerParser):
   def __init__(self, filePath):
      super(ExtendedBundlerOut, self).__init__(filePath)

   def parseConstraints(self, file, numConstraints):
      for cstrIndex in range(numConstraints):
         line = file.readline()
         ref, rel = map(lambda x : float(x), line.split())
         self.constraints.append((ref,rel))

   def parseCams(self, file, numCams):
      for camIndex in range(numCams):
         fx, fy = split_float(file.readline())
         cx, cy = split_float(file.readline())
         k1, k2, k3 = split_float(file.readline())
         
         # Parse R matrix
         R = []
         for i in range(3):
            line = file.readline()            
            R.append(map(lambda x : float(x), line.split()))
            
         line = file.readline()
         t = map(lambda x : float(x), line.split())
         
         intrinsics = vtk.Intrinsics(f=(fx,fy), c=(cx,cy), disto_coeffs=vtk.camera.DistorsionCoeffients((k1,k2,k3)))
         extrinsics = vtk.Extrinsics(R=R, t=t)

         self.cameraList.append(ExtendedBundlerCamera(intrinsics, extrinsics))
 
def parse_bundle_out(filename):
   with open(filename) as f:
      header = f.readline()

   if header.startswith('# Extended Bundle'):
      return ExtendedBundlerOut(filename)
   elif header.startswith('# Bundle'):
      return BundlerOut(filename)
   else:
      raise RuntimeError("Unkown file type.")

def split_float(string):
   return map(lambda x : float(x), string.split())

def point2track(point, id=None):
   views = point.viewList
   cams = map(lambda v: v.camera, views)
   idxes = map(lambda v: v.key, views)
   track = vtk.Track(cams, idxes, id=id, pos=point.pos, color=point.color)
   return track

def bundlerToSiftCoord(bundler, size):
   # WARNING : SIFT uses (row, column) corrdinates.
   middle = (size[0]/2.0, size[1]/2.0)
   
   column = bundler[0] + middle[0]
   row = -bundler[1] + middle[1]
   
   return (row, column)

def bundlerToPILCoord(bundler, size):
   siftCoord = bundlerToSiftCoord(bundler, size)
   return sift.siftToPILCoord(siftCoord)
     
def pointToPixel(point, camera):
   P = projectPoint(point, camera)   
   return P[0:2]

# @TODO : generalize to do in 4D.
def projectPoint(point, camera):
   point = np.array(point.pos)
   
   P = np.dot(np.array(camera.R), point) + np.array(camera.t)   
   disparity = -1.0 / P[2]
   P = -P / P[2]
   norm2 = np.linalg.norm(P)**2
   r = 1.0 + camera.k1 * norm2 + camera.k2 * norm2**2
   P = camera.f * r * P
   
   P.resize(4)
   P[2] = 1.0
   P[3] = disparity * camera.f
   
   return P

def write_constraints(file, constraints):
   for cstr in constraints:
      file.write('%i %i\n' % cstr)

def write_cameras(file, cam_list):
   for cam in cam_list:
      file.write("%f %f\n" % cam.f)
      file.write("%f %f\n" % tuple(cam.principal_point))
      file.write("%f %f %f\n" % cam.disto_coeffs)
      np.savetxt(file, cam.R)
      np.savetxt(file, cam.t.T)

def write_viewlist(file, view_list, cam_list, camID_map):
   file.write("%i" % len(view_list))
   for view in view_list:
      file.write(" %i %i" % (camID_map[view.camera], view.key))
      file.write(" %f %f" % (view.x, view.y))
   file.write("\n")

def write_points(file, point_list, cam_list, camID_map):
   for pt in point_list:
      np.savetxt(file, np.array(pt.pos).reshape((1,3)))
      np.savetxt(file, np.array(pt.color).reshape((1,3)), "%i")
      write_viewlist(file, pt.viewList, cam_list, camID_map)

def identity_map(cam_list):
   return range(len(cam_list))

def write_extended_bundler(filename, cam_list, point_list, constraints=None, camID_map=None):
   if not camID_map:
      camID_map = identity_map(cam_list)

   if not constraints:
      constraints = []

   with open(filename, 'w') as f:
      f.write('# Extended Bundle file v0.1\n')
      f.write('%i %i %i\n' % (len(cam_list), len(point_list), len(constraints)))
      write_constraints(f, constraints)
      write_cameras(f, cam_list)
      write_points(f, point_list, cam_list, camID_map)

def find_view(view_list, cam_idx):
   return [view for view in view_list if view.camera == cam_idx]

if __name__ == "__main__":
   BundlerOut("bundle.out")
