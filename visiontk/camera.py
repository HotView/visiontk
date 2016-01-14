# -*- coding: utf-8 -*-

import numpy as np
import exceptions
from collections import namedtuple, Iterable

class Camera(object):
	""" Projective camera.

	The camera is either parameterized by its projection matrix (P) or 
	by a set of intrinsics and extrinsics parameters.
	"""
	def __init__(self, intrinsics=None, extrinsics=None, P=None):
		"""Initialize the camera parameters.

		Keyword arguments:
		intrinsics -- The intrinsic parameters.
		extrinsics -- The extrinsic parameters.
		P -- The projection matrix.

		Normally, the camera should be initialized either with the projection 
		matrix or with its intrinsics and extrinsics. If both are defined, the 
		extrinsics and intrinsics must result in the same projection matrix as 
		the one given. 
		"""
		self.id = None
		self.intrinsics = intrinsics
		self.extrinsics = extrinsics
		self._P = P

		if (P is not None):
			if (intrinsics is not None and extrinsics is not None):
				tmp_P = projection_matrix(intrinsics.K, extrinsics.Rt)
				assert np.allclose(self.P, tmp_P), "Intrinsics, extrinsics and the projection matrix are inconsistent"

	@property 
	def P(self):
		if (self._P is not None):
			return self._P
		else:
			K = np.matrix(self.intrinsics.K)
			Rt = np.matrix(self.extrinsics.Rt)
			P = K * Rt
			return np.array(P)

	@property 
	def C(self):
		return self.extrinsics.C

	def project(self, point3d):
		raise NotImplementedError()

	def __str__(self):
		np.set_printoptions(suppress=True)
		return "ID:%s\n%s\n%s" % (self.id, str(self.intrinsics.K), str(self.extrinsics.Rt))

	def __repr__(self):
		return self.__str__()


def build_identity_camera():
	return Camera(K=np.eye(3), R=np.eye(3), t=np.zeros(3))


def projection_matrix(K, Rt):
	K = np.matrix(K)
	Rt = np.matrix(Rt)
	P = K * Rt
	return np.array(P)


class Intrinsics(object):
	def __init__(self, f=None, c=None, K=None, disto_coeffs=None):
			self.id = None

			if (K is None):
				if (f is None):
					raise exceptions.ValueError('Focal length must be defined.')
				if (c is None):
					raise exeptions.ValueError('Principal point must be defined.')

				if not isinstance(f, Iterable):
					f = (f,)

				self._K = build_K(f, c)
			else:
				self._K = K

			if (disto_coeffs is None):
				self.disto_coeffs = DistorsionCoeffients((0.0,0.0,0.0,0.0))
			else:
				self.disto_coeffs = disto_coeffs

	def scale(self, factor):
		scale_matrix = np.eye(3)
		scale_matrix[0,0] = factor
		scale_matrix[1,1] = factor
		self._K = np.dot(scale_matrix,self._K)

	@property
	def f(self):
		return (self._K[0,0], self._K[1,1])

	@property
	def c(self):
		return self._K[0:2,2]

	@c.setter
	def c(self, value):
		self._K[0:2,2] = value

	@property
	def K(self):
		return np.array(self._K)

	@K.setter
	def K(self, value):
		self._K = np.array(value)

	def __str__(self):
		return "%s" % (str(self._K))


class DistorsionCoeffients(object):
	def __init__(self, *args):
		self.coeffs = None

		if len(args) == 2 :
			# seperated radial and tangential coefficients
			radial = args[0]
			tangential = args[1]

			# OpenCV format (r1,r2,t1,t2,[r3,r4,r5,r6])
			self.coeffs = np.array(radial[0:2] + tangential + radial[2:])
		elif len(args) == 1 :
			# all coefficients in a single vector
			self.coeffs = np.array(args[0])
		else:
			raise ValueError('Invalid number of parameters.')

	@property
	def radial(self):
		r12 = list(self.coeffs[0:2])
		rRest = list(self.coeffs[4:])
		return tuple(r12 + rRest)

	@radial.setter
	def radial(self, k):
		n = len(k)
		if n < 2 : raise ValueError("Radial coefficients vector must have at least 2 elements.")
		if n < 3 :
			self.coeffs[0:2] = k
		else :
			self.coeffs.resize(2+n)
			self.coeffs[0:2] = k[0:2]
			self.coeffs[4:] = k[2:]

	@property
	def tangential(self):
		return tuple(self.coeffs[2:4])

	def __getitem__(self, val):
		""" Acces the coefficients as if the were stored in a tuple like (radial,tangential).
		For backwards compatibility only! 
		"""
		r = self.radial
		t = self.tangential
		tmp = (r,t)
		return tmp[val]

def build_K(f, c):
	if len(f) == 1:
		f = (f[0],f[0])

	K = np.eye(3)
	K[0,0] = f[0]
	K[1,1] = f[1]
	K[0,2] = c[0]
	K[1,2] = c[1]

	return K

class Extrinsics(object):
	def __init__(self, R=None, t=None):
		if (R is None):
			raise exceptions.ValueError('Rotation matrix must be defined.')
		if (t is None):
			raise exceptions.ValueError('Translation vector must be defined')

		self.R = np.array(R)
		self.t = np.array(t)
		self.t = self.t.reshape(3,1)

	@property
	def Rt(self):
		return np.concatenate((self.R, self.t), axis=1)

	@property 
	def C(self):
		"""Return the camera center"""
		R = np.matrix(self.R)
		t = np.matrix(self.t)
		C = -R.T * t
		return np.array(C)

	def __repr__(self):
		return "%s" % (str(self.Rt))

	def __str__(self):
		return self.__repr__()

def _homogenize(Rt):
	return np.concatenate((Rt, np.array([[0.0,0.0,0.0,1.0]])), axis=0)

def toExtrinsics(Rt):
	R = Rt[0:3,0:3]
	t = Rt[0:3,3]
	return Extrinsics(R=R, t=t)

def inverse_pose(extrinsics):
	R = extrinsics.R
	t = extrinsics.t
	return Extrinsics(R=R.T, t=np.dot(-R.T, t))

def relative_pose(ref_cam, rel_cam):
	rp = np.dot(_homogenize(rel_cam.extrinsics.Rt), _homogenize(inverse_pose(ref_cam.extrinsics).Rt))
	return toExtrinsics(rp)
