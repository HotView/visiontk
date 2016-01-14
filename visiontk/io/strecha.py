import numpy as np
from visiontk import Camera, Intrinsics, Extrinsics

def read_camera(filename):
   with open(filename) as f:
      # Read intrinsics
      K = (f.readline(), f.readline(), f.readline())
      K = map(parse_Strecha_line, K)
      K = np.array(K)

      # Line with three unknown values
      f.readline()

      # Read rotation matrix
      R = (f.readline(), f.readline(), f.readline())
      R = map(parse_Strecha_line, R)
      R = np.array(R)

      # Read camera position
      c = parse_Strecha_line(f.readline())
      c = np.array(c)

      # Read image dimensions
      dims = parse_Strecha_line(f.readline())
      dims = np.array(dims, dtype='int')

   R = R.T
   t = np.dot(-R, c)

   Rt = np.hstack((R,t[:,np.newaxis]))

   intrinsics = Intrinsics(K=K)
   intrinsics.img_dimensions = dims

   return Camera(intrinsics, Extrinsics(R=R, t=t))

def parse_Strecha_line(line):
   return map(float, line.split())