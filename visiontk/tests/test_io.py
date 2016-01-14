import os.path
import unittest

from config import DATA_PATH
import visiontk
from visiontk.keypoint import Keypoint


class TestSIFT(unittest.TestCase):
   def setUp(self):
      self.keypoints = visiontk.io.read_SIFT(os.path.join(DATA_PATH, 'sift.key.gz'))
      self.last_kp = Keypoint((2.480000, 716.360000), 
                              1.000000,
                              -0.620000,
                              [0, 0, 0, 0, 0, 0, 0, 0, 18, 6, 0, 0, 0, 0, 0, 0, 128, 55, 0, 0, 0, 0, 0, 0, 160,
                               39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 92, 23, 0, 0, 0, 2, 7, 4, 173, 61,
                               0, 0, 0, 0, 0, 4, 173, 28, 0, 0, 0, 0, 0, 6, 4, 3, 3, 0, 0, 2, 4, 3, 81, 6, 0,
                               0, 0, 1, 17, 35, 173, 12, 0, 0, 0, 0, 1, 50, 173, 8, 0, 0, 0, 0, 0, 21, 5, 18,
                               4, 0, 0, 0, 0, 0, 76, 31, 0, 0, 0, 0, 0, 3, 173, 25, 0, 0, 0, 0, 0, 19, 173, 8,
                               0, 0, 0, 0, 0, 13])
      
   def test_parsing(self):
      self.assertEqual(len(self.keypoints), 2013)      
      self.assertEqual(self.keypoints[-1], self.last_kp)


class TestBundler(unittest.TestCase):
   def setUp(self):
      pass

   def test_parsing(self):
      self.out = visiontk.io.BundlerOut(os.path.join(DATA_PATH, 'bundler.out'))


if __name__ == "__main__":
   unittest.main()
   