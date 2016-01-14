
import sys
import numpy as np
from scipy.linalg import rq

import cv2

from visiontk import Extrinsics


def find_essential_matrix(F, (intrinsics1, intrinsics2)):
	return np.dot(intrinsics2.K.T, np.dot(F, intrinsics1.K))

def dehomogenize(pt):
	tmp = pt / pt[-1]
	return tmp[0:-1]

def find_extrinsics(E, points):
	""" Find the extrinsics parameters (R,t) between two calibrated cameras based on their Essential matrix.
	    E : The essential matrix.
	    points : A pair of matching points between the two views. The points must be in a normalized camera, i.e. before applying the intrinsics parameters."""
	W = np.array([[0,-1,0],
		          [1,0,0],
		          [0,0,1]])

	U, S, V = np.linalg.svd(E)

	u3 = U[:,2]

	Ra = np.dot(U, np.dot(W, V))
	Rb = np.dot(U, np.dot(W.T, V))

	if np.linalg.det(Ra) < 0 :
		Ra = Ra * -1
	if np.linalg.det(Rb) < 0 :
		Rb = Rb * -1
	
	t_list = [u3, -u3, u3, -u3]
	R_list = [Ra, Ra, Rb, Rb]
	P_list = map(lambda R,t: np.hstack((R, t.reshape(3,1))), R_list, t_list)

	PI = np.array([[1,0,0,0],
		           [0,1,0,0],
		           [0,0,1,0]])

	valid_configs = []

	for i,(R,t,P) in enumerate(zip(R_list, t_list, P_list)):
		pts3D_1 = triangulate_point(PI, P, np.array(points[0]), np.array(points[1]))
		pts3D_1 = dehomogenize(pts3D_1)
		pts3D_1 = np.reshape(pts3D_1, (3,))
		pts3D_2 = np.dot(R, pts3D_1) + t

		# The 3D points must be in front of both cameras, i.e. have a positive Z.
		if pts3D_1[2] > 0 and pts3D_2[2] > 0:
			valid_configs.append(i)

	if len(valid_configs) != 1:
		raise RuntimeError('No valid extrinsic parameters (R,t) found.')
	
	k = valid_configs[0]
	return Extrinsics(R=R_list[k], t=t_list[k])

def absolute_extrinsics(ref_extrinsics, relative_extrinsics):
	Rt_ref = ref_extrinsics.Rt
	Rt_relative = relative_extrinsics.Rt
	row = np.array([0.,0.,0.,1.])
	Rt_ref = np.vstack((Rt_ref,row))
	Rt_relative = np.vstack((Rt_relative, row))

	Rt_absolute = np.dot(Rt_relative, Rt_ref)
	R = Rt_absolute[0:3, 0:3]
	t = Rt_absolute[0:3, -1]

	return Extrinsics(R=R, t=t)

def null(A):
   u, s, vh = np.linalg.svd(A)
   return vh[-1,:]

def skew(v):
    M = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return M

def extract_fundamental((P0, P1)):
    C = null(P0)
    C = C / C[-1]

    e = np.dot(P1, C)
    e_skew = skew(e)

    tmp = np.dot(P1, np.linalg.pinv(P0))
    F = np.dot(e_skew, tmp)
    return F

def triangulate_point(P0, P1, pt0, pt1):
	x0 = pt0[0]
	y0 = pt0[1]
	x1 = pt1[0]
	y1 = pt1[1]

	A = np.array([x0*P0[2,:] - P0[0,:],
	              y0*P0[2,:] - P0[1,:],
	              #x0*P0[1,:] - y0*P0[0,:],
	 	          x1*P1[2,:] - P1[0,:],
	 	          y1*P1[2,:] - P1[1,:]])
	 	          #x1*P1[1,:] - y1*P1[0,:]])

	U,S,V = np.linalg.svd(A)
	X = V[-1,:]

	return X / X[-1]

def iterative_point_triangulation(P0, P1, pt0, pt1):
	w0 = 1
	w1 = 1

	x0 = pt0[0]
	y0 = pt0[1]
	x1 = pt1[0]
	y1 = pt1[1]

	# DLT triangulation
	X = triangulate_point(P0, P1, pt0, pt1)

	for i in range(10):
		# Measure weights based on reprojection error
		w0 = np.dot(P0[2,:], X)
		w1 = np.dot(P1[2,:], X)

		# Solve with weights
		A = np.array([x0*P0[2,:] - P0[0,:],
		          y0*P0[2,:] - P0[1,:],
		          x1*P1[2,:] - P1[0,:],
		          y1*P1[2,:] - P1[1,:]])

		A[0,:] = A[0,:] * (1/w0)
		A[1,:] = A[1,:] * (1/w0)
		A[2,:] = A[2,:] * (1/w1)
		A[3,:] = A[3,:] * (1/w1)

		B = -A[:,-1]
		A = A[:,0:3]

		tmp, X = cv2.solve(A,B, flags=cv2.DECOMP_SVD)

		X.reshape(3)
		X = np.append(X,1)

	return X

def triangulate_points(P0, P1, pts0, pts1, algo='ITERATIVE'):
	assert pts0.shape == pts1.shape

	if algo == 'ITERATIVE':
		return np.array(map(lambda x,y: iterative_point_triangulation(P0, P1, x, y), pts0.T, pts1.T)).T
	elif algo == 'LINEAR':
		return np.array(map(lambda x,y: triangulate_point(P0, P1, x, y), pts0.T, pts1.T)).T
	else:
		raise ValueError('Invalid triangulation algorithm.')

def build_DLT_matrix(pts3D, pts2D):
	m = np.empty((pts3D.shape[0]*2,12))

	for i, (pt3D, pt2D) in enumerate(zip(pts3D, pts2D)):
		X = pt3D[0]
		Y = pt3D[1]
		Z = pt3D[2]
		u = pt2D[0]
		v = pt2D[1]

		m[2*i,:]   = np.array([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])
		m[2*i+1,:] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])

	return m

def DLT(pts3D, pts2D):
	m = build_DLT_matrix(pts3D, pts2D)
	U,W,V = np.linalg.svd(m)
	P = V[-1,:].reshape((3,4))
	return P

def extract_KRt(P):
	"""Based on http://www.janeriksolem.net/2011/03/rq-factorization-of-camera-matrices.html"""
	H = P[0:3, 0:3]
	K, R = rq(H)
	T = np.diag(np.sign(np.diag(K)))
	K = np.dot(K,T)
	R = np.dot(T,R)
	t = np.dot(np.linalg.inv(K), P[:,-1])

	if np.linalg.det(R) < 0 :
		R = np.dot(R, -1 * np.eye(3))
		t = np.dot(-1 * np.eye(3), t)

	K = K/K[-1,-1]

	return K,R,t

def solve_rigid(Z, X):
	""" Find the rigid transformation T mapping the 3D points Z to the 3D points X. """
	assert(X.shape == Z.shape)
	assert(Z.shape[1] == 3)

	N = Z.shape[0]

	Zmean = np.mean(Z, axis=0)
	Xmean = np.mean(X, axis=0)

	Zc = Z - Zmean
	Xc = X - Xmean

	H = np.dot(Zc.T, Xc)

	U, S, V = np.linalg.svd(H)
	# np.linalg.svd() returns V.T instead of the more usual V.
	V = V.T
	R = np.dot(V, U.T)

	t = Xmean - np.dot(R, Zmean);

	T = R
	T = np.hstack((T, t[:,np.newaxis]))
	T = np.vstack((T, [[0, 0, 0 , 1]]))
	return T

