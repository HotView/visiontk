import tempfile, shutil
import yaml
import numpy as np

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(list(mapping["data"]), dtype=mapping["dt"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

def load_yaml(filename):
	content = None

	with open_yaml(filename) as f:
		content = yaml.load(f.read())

	return content

def open_yaml(filename):
	return OpencvYamlFile(filename)

def bypass_yaml_directive(file):
	file.seek(0)

	# Remove YAML 1.0 if it is present.
	# Necessary since PyYAML supports only YAML 1.1+
	directive = file.readline()

	if ("%YAML:1.0" not in directive):
		file.seek(0)

	return file

class OpencvYamlFile(object):
	def __init__(self, filename):
		self.filename = filename
		self.file = None

	def __enter__(self):
		self.file = open(self.filename)
		return bypass_yaml_directive(self.file)

	def __exit__(self ,type, value, traceback):
		self.file.close()
