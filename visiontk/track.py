import copy
import numpy as np

class Track(object):
	def __init__(self, cams, idx, id=None, pos=None, color=None):
		self.cams = cams
		self.idx = idx
		self.id = id
		self.pos = pos
		self.color = color

	def __getitem__(self, key):
		return self.__dict__[key]

	def __setitem__(self, key, value):
		self.__dict__[key] = value

	def __repr__(self):
		return "{id:%s,cams:%s,idx:%s,pos:%s,color:%s}" % (str(self.id), self.cams, self.idx, self.pos, self.color)

def find_tracks(keypoints, matches):
	matches2 = copy.deepcopy(matches)
	tracks = []

	for m_idx, m in enumerate(matches2):
		# For each matching pairs of cameras
		cam_idx = m["cam_pair"][0]
		for kp_pair in m["idx"]: 
			# For each matching pairs of keypoints in the camera pair
			track = Track([cam_idx], [kp_pair[0]])
			next_kp = kp_pair[1]
			m["idx"].remove(kp_pair) # @TODO : verify if necessary.

			next_match_idx = m_idx + 1
			follow_track(track, next_kp, matches2, next_match_idx)
			tracks.append(track)
	return tracks

def follow_track(track, kp_idx, matches, match_idx):
	if match_idx >= len(matches):
		# Last camera : end of track
		# Must terminate track
		prev_matches = matches[match_idx-1]

		cam_idx = prev_matches["cam_pair"][1]

		track["cams"].append(cam_idx)
		track["idx"].append(kp_idx)
		return track
	else:
		cam_idx = matches[match_idx]["cam_pair"][0]
		matching_kp_pairs = filter(lambda x : x[0] == kp_idx, matches[match_idx]["idx"])
		
		if len(matching_kp_pairs) > 0:
			kp_pair = matching_kp_pairs[0]
			track["cams"].append(cam_idx)
			track["idx"].append(kp_pair[0])

			next_kp = kp_pair[1]
			matches[match_idx]["idx"].remove(kp_pair)

			next_match_idx = match_idx + 1
			follow_track(track, next_kp, matches, next_match_idx)
		else:
			# No more matching keypoint : end of track
			# Must terminate track
			track["cams"].append(cam_idx)
			track["idx"].append(kp_idx)
			return track

def validate_track(track, min_len=3):
	return len(track["cams"]) >= min_len

def identify_tracks(tracks):
	for uid, t in enumerate(tracks):
		t['id'] = uid
	return tracks

def extract_tracks_subset(tracks, cam_list):
	extracted = tracks
	for cam in cam_list:
		extracted = select_tracks(extracted, cam)
	return extracted

def select_tracks(tracks, cam):
	selected = []
	for track in tracks:
		if cam in track['cams']:
			selected.append(track)
	return selected

def select_tracks_cam_idxs(tracks, cam):
	idxs = []
	for track in tracks:
		idxs.append(track['idx'][track['cams'].index(cam)])
	return idxs

def get_track_keypoints(tracks, cam, keypoints):
	idxs = select_tracks_cam_idxs(tracks, cam)
	return map(lambda i : keypoints['keypoints'][i].pt, idxs)

def number_keypoints_in_tracks(tracks, cam_idx):
	n_per_track = [len(set(t["cams"]).intersection(set(cam_idx))) for t in tracks]
	return  reduce(lambda x,y:x+y, n_per_track)

def assign_positions(tracks, points3D):
	assert(len(tracks) == len(points3D))
	for t,pos in zip(tracks, points3D):
		t["pos"] = pos
	return tracks

def get_positions(tracks):
	points3D = np.empty((len(tracks),3))
	for i, t in enumerate(tracks):
		points3D[i,:] = t["pos"]
	return points3D

def has_position(track):
	return track["pos"] is not None

def triangulated_tracks(tracks):
	return [t for t in tracks if has_position(t)]

def untriangulated_tracks(tracks):
	return [t for t in tracks if not has_position(t)]