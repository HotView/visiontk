import numpy as np

def find_similarity(points, ref):
	scale, origin = find_scale(points, ref)
	points = apply_scale(points, scale, origin)
	R, t = find_rigid(points, ref)
	return scale, origin, R, t

def apply_similarity(points, scale, origin, R, t):
	results = apply_scale(points, scale, origin)
	results = apply_rigid(results, R, t)
	return results

def centroid(pts):
	return np.mean(pts, axis=0)

def find_rigid(points, ref):
	""" http://nghiaho.com/?page_id=671 """
	points_centroid = centroid(points)
	ref_centroid = centroid(ref)

	points_ = points - points_centroid
	ref_ = ref - ref_centroid
	
	H = np.dot(points_.T, ref_)

	U,S,V = np.linalg.svd(H)
	R = np.dot(V.T, U.T)

	t = np.dot(-R, points_centroid) + ref_centroid

	return R, t

def apply_rigid(points, R, t):
	return np.dot(R, points.T).T + t

def mean_norm(pts, origin=None):
	if origin is None:
		origin = np.zeros(3)

	pts_ = pts - origin
	pts_ = np.array(map(np.linalg.norm, pts_))
	return np.mean(pts_)

def find_scale(points, ref, origin_idx=None):
	""" origin_idx : the point (match) to use as the reference point. If None, the reference point is the mean """
	if origin_idx is None:
		origin_pts = np.mean(points, 0)
		origin_ref = np.mean(ref, 0)
	else:
		origin_pts = points[origin_idx,:]
		origin_ref = ref[origin_idx,:]

	points_ = mean_norm(points, origin_pts)
	ref_ = mean_norm(ref, origin_ref)

	scale = ref_ / points_

	return scale, origin_pts

def apply_scale(pts, scale, origin):
	return (pts - origin)*scale + origin 