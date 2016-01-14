
import numpy as np

def bundler_to_openCV_2D(pts, img_dimensions):
	""" pts : Array of points Nx2. One row per point."""
	pts = normalize_points_shape(pts, 2)

	converted_pts = np.empty(pts.shape, pts.dtype)
	converted_pts[:,0] = pts[:,0]
	converted_pts[:,1] = -pts[:,1]
	converted_pts = converted_pts + (img_dimensions / 2.0)

	return converted_pts

def openCV_to_bundler_2D(pts, img_dimensions):
	""" pts : Array of points Nx2. One row per point."""
	pts = normalize_points_shape(pts, 2)
	
	converted_pts = np.empty(pts.shape, pts.dtype)
	converted_pts = pts - (img_dimensions / 2.0)
	converted_pts[:,1] = -converted_pts[:,1]

	return converted_pts

def openCV_to_extended_bundler_2D(pts, img_dimensions):
	""" pts : Array of points Nx2. One row per point."""
	pts = normalize_points_shape(pts, 2)
	
	converted_pts = np.empty(pts.shape, pts.dtype)
	converted_pts = pts

	return converted_pts

def bundler_to_strecha_2D(pts, img_dimensions):
	""" pts : Array of points Nx2. One row per point."""
	pts = normalize_points_shape(pts, 2)

	converted_pts = np.empty(pts.shape, pts.dtype)
	converted_pts = -pts
	converted_pts = converted_pts + (img_dimensions / 2.0)

	return converted_pts

def normalize_points_shape(pts, length):
	normalized = pts
	if len(pts.shape) == 1:
		assert pts.shape[0] == length
		normalized = np.copy(pts)
		normalized = normalized.reshape((1,length))
	else :
		assert len(pts.shape) == 2
		assert pts.shape[1] == length

	return normalized
