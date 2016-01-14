import numpy as np
from visiontk import Camera, Intrinsics, Extrinsics

def writePLY(ply_output, points, colors):     
   file = open(ply_output, "w")

   file.write("ply\n")
   file.write("format ascii 1.0\n")
   file.write("element vertex %i\n" % len(points))
   file.write("property float x\n")
   file.write("property float y\n")
   file.write("property float z\n")
   file.write("property uchar diffuse_red\n")
   file.write("property uchar diffuse_green\n")
   file.write("property uchar diffuse_blue\n")
   file.write("end_header\n")

   file.writelines("%g %g %g %i %i %i\n" % (point[0], point[1], point[2], 
                                            color[0], color[1], color[2]) 
                   for (point,color) in zip(points,colors))
   file.close()

def readPLY_vertex(ply_input):
   with open(ply_input, 'r') as file:
      elements = read_PLY_header(file)

      # TODO : handle case when vertex is not the first element
      assert(elements[0]["type"] == "vertex")
      
      length = elements[0]["length"]
      width = 3 # TODO : we should not assert that the width is 3 or that the first 3 properties are respectively x,y,z.
      vertex = np.empty((length,width), dtype='float')

      for i in xrange(length):
         position = [float(x) for x in file.readline().rstrip().split()[0:3]]
         vertex[i,:] = np.array(position, dtype='float')
   return vertex

def read_PLY_header(f):
   magic_string = f.readline().rstrip()
   assert(magic_string == "ply")
   format_string = f.readline().rstrip()

   line = ""
   lines = []
   while line != "end_header":
      line = f.readline().rstrip()
      lines.append(line)

   elements = extract_elements(lines[:-1])
   return elements

def extract_elements(lines):
   elements = []

   for line in lines:
      tokens = line.split()

      if tokens[0] == "element":
         elements.append({"type" : tokens[1],
                          "properties" : [],
                          "length" : int(tokens[2])}) 
      elif tokens[0] == "property":
         elements[-1]["properties"].append(' '.join(tokens[1:]))
   return elements