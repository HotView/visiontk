from __future__ import print_function

import collections
import itertools
import numpy as np

class World(object):
	def __init__(self, cameras=None, points=None, projections=None, constraints=None):
		self.cameras = cameras
		self.points = points
		self.projections = projections
		self._constraints = []
		self._tracks = None

	def view(self, cameraIDs=None):
		return WorldView(self, cameraIDs)

	def add_bundler_points(self, bundler_points):
		self.points = []
		self.projections = ProjectionStore()

		for i, bp in enumerate(bundler_points):
			self.points.append(Point(bp.pos, bp.color))
			for view in bp.viewList :
				projID = ProjectionID(view.camera, i)
				proj = Projection(view.position, view.camera, i, view.key)
				self.projections[projID] = proj

		self._build_tracks()

	def remove_unselected_points(self, selected_pointIDs):
		selected_pointIDs = set(selected_pointIDs)
		
		new_ID_map = {}
		for nid, pid in enumerate(selected_pointIDs):
			new_ID_map[pid] = nid

		self.points = [self.points[i] for i in selected_pointIDs]

		selected_projections = []
		for (projID, proj) in self.projections.projections.iteritems():
			if projID.pointID in selected_pointIDs:
				proj.pointID = new_ID_map[proj.pointID]
				selected_projections.append(proj)
		self.projections = ProjectionStore(selected_projections)

		self._build_tracks()

	def remove_unselected_cameras(self, selected_camIDs):
		selected_camIDs = set(selected_camIDs)

		new_ID_map = {}
		for nid, pid in enumerate(selected_camIDs):
			new_ID_map[pid] = nid

		self.cameras = [self.cameras[i] for i in selected_camIDs]

		new_constraints = []
		for cstr in self._constraints:
			new_constraints.append(tuple(map(lambda id : new_ID_map[id], cstr)))
		self._constraints = new_constraints

		selected_projections = []
		for (projID, proj) in self.projections.projections.iteritems():
			if projID.cameraID in selected_camIDs:
				proj.cameraID = new_ID_map[proj.cameraID]
				selected_projections.append(proj)
		self.projections = ProjectionStore(selected_projections)

		self._build_tracks()

	def mean_reproj_error(self):
		proj_count = len(self.projections)
		total_error = 0.0
		for id, proj in self.projections.iteritems():
			cam = self.cameras[id.cameraID]
			pt = self.points[id.pointID]
			total_error += np.linalg.norm(proj.position - cam.project(pt.position))
		return total_error/proj_count

	@property
	def constraints(self):
		return self._constraints

	@constraints.setter
	def constraints(self, constraints):
		self._constraints = constraints

	def _build_tracks(self):
		self._tracks = TrackStore()
		for pointID in xrange(len(self.points)):
			if pointID in self.projections.point_index:
				projectionIDs = list(self.projections.point_index[pointID])
				self._tracks[pointID] = Track(pointID, projectionIDs)

	def __str__(self):
		return "[%s cameras, %s points, %s projections]" % (len(self.cameras), len(self.points), len(self.projections))

class ProjectionStore(object):
	def __init__(self, projections=None):
		self.projections = {}
		self.camera_index = {}
		self.point_index = {}
		self.visibility_map = {}

		if projections is not None:
			for proj in projections:
				self[ProjectionID(proj.cameraID, proj.pointID)] = proj

	def __getitem__(self, projID):
		return self.projections[projID]

	def __setitem__(self, projID, proj):
		self.projections[projID] = proj

		if projID.cameraID not in self.camera_index:
			self.camera_index[projID.cameraID] = set((projID,))
		else :
			self.camera_index[projID.cameraID].add(projID)
		
		if projID.pointID not in self.point_index:
			self.point_index[projID.pointID] = set((projID,))
		else :
			self.point_index[projID.pointID].add(projID)

		if projID.cameraID not in self.visibility_map:
			self.visibility_map[projID.cameraID] = set((projID.pointID,))
		else :
			self.visibility_map[projID.cameraID].add(projID.pointID)

	def __repr__(self):
		for ID, proj in self.projections.iteritems():
			print("%s : %s"  % (repr(ID), repr(proj)))

	def __len__(self):
		return len(self.projections)

	def iterkeys(self):
		return self.projections.iterkeys()

	def iteritems(self):
		return self.projections.iteritems()

	def select_cameraIDs(self, cameraIDs):
		""" Returns the subset of ProjectionIDs with Points projected in all cameraIDs """
		if not isinstance(cameraIDs, collections.Iterable):
			cameraIDs = [cameraIDs,]

		pointIDs = self.common_cameras(cameraIDs)
		projIDs = itertools.product(cameraIDs, pointIDs)
		return set([ProjectionID(*projID) for projID in projIDs])

	def common_cameras(self, cameraIDs):
		""" Returns the ID of the points visible in all cameraIDs """
		cameraIDs = list(cameraIDs)
		# Must copy point ID set, otherwise the intesection operator is applied to the set in the visibility map.
		pointIDs = set(self.visibility_map[cameraIDs[0]])
		for cameraID in cameraIDs[1:]:
			pointIDs &= self.visibility_map[cameraID]
		return pointIDs

	def select_common(self, pointID, cameraIDs):
		return self.select_cameraIDs(cameraIDs) & self.point_index[pointID]

class TrackStore(object):
	def __init__(self):
		self._track_map = {}

	def __getitem__(self, pointID):
		return self._track_map[pointID]

	def __setitem__(self, pointID, track):
		self._track_map[pointID] = track

	def __repr__(self):
		for pointID, track in self._track_map.iteritems():
			print("%s : %s"  % (pointID, repr(track)))

	def __len__(self):
		return len(self._track_map)

class WorldView(object):
	""" Gives a view on the objects of a World based on a subset of cameras.
	    The view selects 3D points visible in all the camera subset. """
	def __init__(self, world, cameraIDs=None):
		self.world = world

		if cameraIDs :
			self.cameraIDs = set(cameraIDs)
		else :
			self.cameraIDs = world.cameras.viewkeys()

	@property
	def cameras(self):
		return (self.world.cameras[camID] for camID in self.cameraIDs)

	@property
	def points(self):
		return (self.world.points[projID.pointID] for projID in self.projectionIDs)

	@property
	def pointIDs(self):
		return (projID.pointID for projID in self.projectionIDs)

	@property
	def projections(self):
		return (self.world.projections[projID] for projID in self.projectionIDs)

	@property
	def projectionIDs(self):
		return self.world.projections.select_cameraIDs(self.cameraIDs)

	@property
	def tracks(self):
		"""Returns list of tracks in current view. The track order is undefined."""
		return [self.world._tracks[pointID] for pointID in self.world.projections.common_cameras(self.cameraIDs)]
		
	def extract_projections_position(self, camID):
		"""Return an array of the world view projected points in camera camID.
		Points are ordered according to their associated 3D point ID."""
		tracks = sorted(self.tracks, key=lambda t : t.pointID)
		points2D = np.empty((len(tracks),2))
		pointsID = np.empty(len(tracks), dtype='int')

		for i, track in enumerate(tracks):
			projID = track.select_camera(camID)
			assert(projID.cameraID == camID)
			points2D[i,:] = self.world.projections[projID].position
			pointsID[i] = projID.pointID

		return (pointsID,points2D)

ProjectionID = collections.namedtuple('ProjectionID', ['cameraID','pointID'])

class Projection(object):
	def __init__(self, position, cameraID, pointID, key=None):
		self.position = position
		self.key = key
		self.cameraID = cameraID
		self.pointID = pointID

	def __repr__(self):
		return "(position=%s,key=%s,cameraID=%s,pointID=%s)" % (self.position, self.key, self.cameraID, self.pointID)

class Point(object):
	def __init__(self, position, color=None):
		self.position = position
		self.color = color

	def __repr__(self):
		return "position:%s color:%s" % (self.position, self.color)

class Track(object):
	def __init__(self, pointID, projectionIDs):
		self.pointID = pointID
		self._projectionIDs = list(projectionIDs) # _projectionIDs must be ordered for backwards compatibility
		self.camera_index = self._build_camera_index(projectionIDs)

	def _build_camera_index(self, projectionIDs):
		camera_index = {}
		for projID in projectionIDs:
			camera_index[projID.cameraID] = projID
		return camera_index

	def __repr__(self):
		return 'Track(pointID:%s, projections:%s)' % (self.pointID, self._projectionIDs)

	@property
	def projectionIDs(self):
	    return self._projectionIDs
	
	@projectionIDs.setter
	def projectionIDs(self, value):
	    self._projectionIDs = value
	    self.camera_index = self._build_camera_index(value)

	def select_camera(self, cameraID):
		for id in self._projectionIDs:
			if id.cameraID == cameraID:
				return id
		raise KeyError("Point %i not projected in camera %i" % (self.pointID, cameraID))
