import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def read_file(filename):
	with open(filename, 'r') as fin:
		text = fin.readlines()
	return text

def parse_data(data):
	num_shots = len(data[2].split())

	d = {}
	for i in range(num_shots):
		d["shot{}".format(i+1)] = []
	
	for line in data[2:]:
		for j in range(num_shots):
			d["shot{}".format(j+1)].append(float(line.split()[j]))

	for i in range(num_shots):
		if all(v == 0 for v in d["shot{}".format(i+1)]):
			print('shot{} deleted'.format(i+1))
			d.pop("shot{}".format(i+1))
		else:
			while d["shot{}".format(i+1)][-1] == 0:
				d["shot{}".format(i+1)].pop()
	return d


def read_subject(i):
	accelx = parse_data(read_file('DataCollection/Subject{}/accelx.txt'.format(i)))
	accely = parse_data(read_file('DataCollection/Subject{}/accely.txt'.format(i)))
	accelz = parse_data(read_file('DataCollection/Subject{}/accelz.txt'.format(i)))
	#emg1 = parse_data(read_file('DataCollection/Subject{}/emg1.txt'.format(i)))
	#emg2 = parse_data(read_file('DataCollection/Subject{}/emg2.txt'.format(i)))
	# emg3 = parse_data(read_file('DataCollection/Subject{}/emg3.txt'.format(i)))
	#emg4 = parse_data(read_file('DataCollection/Subject{}/emg4.txt'.format(i)))
	# emg5 = parse_data(read_file('DataCollection/Subject{}/emg5.txt'.format(i)))
	#emg6 = parse_data(read_file('DataCollection/Subject{}/emg6.txt'.format(i)))
	#emg7 = parse_data(read_file('DataCollection/Subject{}/emg7.txt'.format(i)))
	#emg8 = parse_data(read_file('DataCollection/Subject{}/emg8.txt'.format(i)))
	# emgtimes = parse_data(read_file('DataCollection/Subject{}/emgtimes.txt'.format(i)))
	gyrox = parse_data(read_file('DataCollection/Subject{}/gyrox.txt'.format(i)))
	gyroy = parse_data(read_file('DataCollection/Subject{}/gyroy.txt'.format(i)))
	gyroz = parse_data(read_file('DataCollection/Subject{}/gyroz.txt'.format(i)))
	imutimes = parse_data(read_file('DataCollection/Subject{}/imutimes.txt'.format(i)))
	shootingaccuracy = read_file('DataCollection/Subject{}/shootingaccuracy.txt'.format(i))
	subjectinfo = read_file('DataCollection/Subject{}/subjectinfo.txt'.format(i))
	quat1 = parse_data(read_file('DataCollection/Subject{}/quat1.txt'.format(i)))
	quat2 = parse_data(read_file('DataCollection/Subject{}/quat2.txt'.format(i)))
	quat3 = parse_data(read_file('DataCollection/Subject{}/quat3.txt'.format(i)))
	quat4 = parse_data(read_file('DataCollection/Subject{}/quat4.txt'.format(i)))
	# rot1 = parse_data(read_file('DataCollection/Subject{}/rot1.txt'.format(i)))
	# rot2 = parse_data(read_file('DataCollection/Subject{}/rot2.txt'.format(i)))
	# rot3 = parse_data(read_file('DataCollection/Subject{}/rot3.txt'.format(i)))


	shots = len(accelx)

	d = {}
	d['subjectinfo'] = subjectinfo
	d['numshots'] = shots

	for j in range(shots):
			d['shot{}'.format(j+1)] = {}
			d['shot{}'.format(j+1)]['make'] = shootingaccuracy[0][j]
			d['shot{}'.format(j+1)]['accel'] = list(zip(accelx["shot{}".format(j+1)],accely["shot{}".format(j+1)],accelz["shot{}".format(j+1)]))
			d['shot{}'.format(j+1)]['quat'] = list(zip(quat1['shot{}'.format(j+1)], quat2['shot{}'.format(j+1)], quat3['shot{}'.format(j+1)], quat4['shot{}'.format(j+1)]))
			d['shot{}'.format(j+1)]['gyro'] = list(zip(gyrox["shot{}".format(j+1)],gyroy["shot{}".format(j+1)],gyroz["shot{}".format(j+1)]))
			#d['shot{}'.format(j+1)]['rot'] = list(zip(rot1["shot{}".format(j+1)],rot2["shot{}".format(j+1)],rot3["shot{}".format(j+1)]))
			d['shot{}'.format(j+1)]['accelx'] = accelx["shot{}".format(j+1)]
			d['shot{}'.format(j+1)]['accely'] = accely["shot{}".format(j+1)]
			d['shot{}'.format(j+1)]['accelz'] = accelz["shot{}".format(j+1)]
			#d['shot{}'.format(j+1)]['emg1'] = emg1["shot{}".format(j+1)]
			#d['shot{}'.format(j+1)]['emg2'] = emg2["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['emg3'] = emg3["shot{}".format(j+1)]
			#d['shot{}'.format(j+1)]['emg4'] = emg4["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['emg5'] = emg5["shot{}".format(j+1)]
			#d['shot{}'.format(j+1)]['emg6'] = emg6["shot{}".format(j+1)]
			#d['shot{}'.format(j+1)]['emg7'] = emg7["shot{}".format(j+1)]
			#d['shot{}'.format(j+1)]['emg8'] = emg8["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['emgtimes'] = emgtimes["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['gyrox'] = gyrox["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['gyroy'] = gyroy["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['gyroz'] = gyroz["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['imutimes'] = imutimes["shot{}".format(j+1)]

	return d


def magnitude(a):
	x,y,z = a
	return math.sqrt(x**2 + y**2 + z**2)

def read_ranges():
	text = read_file()

def quat_vec_at_release(shot):
	i = shot['emg8'].index(max(shot['emg8']))
	j = len(shot['quat'])*i/len(shot['emg1'])
	a,b,c,d = shot['quat'][int(j)]
	return (b,c,d)


def bottom_emg_diff(shot):
	i = shot['emg8'].index(max(shot['emg8']))
	i1 = i-25
	i2 = i+25
	total = 0
	while i1 < i2:
		total += (shot['emg6'][i1]+shot['emg7'][i1]) - (shot['emg1'][i1]+shot['emg2'][i1])
		i1 += 1
	return (total)**2

def print_quats(shot):
	z = [z for x,y,z in shot['accel_norm']]
	idx = z.index(min(z))

	X = []
	Y = []
	Z = []

	for i in range(idx-25,idx+1):
		a,x,y,z = shot['quat'][i]
		X.append(x)
		Y.append(y)
		Z.append(z)

	U = []
	V = []
	W = []
	for i in range(len(X)):
		U.append(i/20)
		V.append(0)
		W.append(0)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	ax.quiver(U,V,W,X,Y,Z)
	ax.set_xlim([-1,10])
	ax.set_ylim([-1,3])
	ax.set_zlim([-1,2])
	plt.show()

def print_rots(shot):
	z = [z for x,y,z in shot['accel_norm']]
	idx = z.index(min(z))

	X = []
	Y = []
	Z = []

	for i in range(idx-25,idx+1):
		x,y,z = shot['rot'][i]
		X.append(x)
		Y.append(y)
		Z.append(z)

	print(X,Y,Z)

def trim_shot(shot):
	z = [z for x,y,z in shot['accel_norm']]
	idx = z.index(min(z))

	v = []
	for i in range(idx-25,idx+1):
		x,y,z = shot['accel_norm'][i]
		v.append((x,y))

	x_avg= sum([x for x,y in v])
	y_avg = sum([y for x,y in v])

	dot_sum = []
	for x,y in v:
		dot_sum.append(abs((x*x_avg)+(y*y_avg)))

	return sum(dot_sum)

def print_shot(shot):
	U = []
	V = []
	W = []
	for i in range(len(shot['accelx'])):
		U.append(i/20)
		V.append(0)
		W.append(0)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	X = [x_ for x_,y_,z_ in shot['accel_norm']]
	Y = [y_ for x_,y_,z_ in shot['accel_norm']]
	Z = [z_ for x_,y_,z_ in shot['accel_norm']]

	M = [magnitude(v) for v in shot['accel_norm']]
	# X = [x_ for x_,y_,z_ in rot_vectors(shot['accel'], shot['quat'])]
	# Y = [y_ for x_,y_,z_ in rot_vectors(shot['accel'], shot['quat'])]
	# Z = [z_ for x_,y_,z_ in rot_vectors(shot['accel'], shot['quat'])]

	ax.quiver(U,V,W,X,Y,Z, M)
	ax.set_xlim([-1,10])
	ax.set_ylim([-1,3])
	ax.set_zlim([-1,2])
	plt.show()		

def rot_vectors(vectors, quats):
	vectors_base = []
	z_normed = []
	for vector, quat in zip(vectors, quats):
		x,y,z = vector
		a,b,c,d = quat
		vx = (a**2+b**2-c**2-d**2)*x + (2*b*c-2*a*d)*y + (2*b*d+2*a*c)*z
		vy = (2*b*c+2*a*d)*x + (a**2-b**2+c**2-d**2)*y + (2*c*d-2*a*b)*z
		vz = (2*b*d-2*a*c)*x + (2*c*d+2*a*b)*y + (a**2-b**2-c**2+d**2)*z

		vectors_base.append((vx,vy,vz-1))
		z_normed.append(vz-1)

	return (vectors_base, z_normed)


d = {}
for i in range(15, 18):
	if i==14: continue
	print(i)
	d['subject{}'.format(i)] = read_subject(i)

shots = {}
i = 1
for subj in d:
	for key in d[subj]:
		if key[0:4] == 'shot':
			shots[i] = d[subj][key]
			i += 1


make = []
Quat_at = []

for key in shots:
 	make.append(shots[key]['make'])

for key in shots:
 	shots[key]['accel_norm'], shots[key]['z_accel_norm'] = rot_vectors(shots[key]['accel'], shots[key]['quat'])

make = []
feat2 = []
feat5 = []


for key in shots:
 	plt.plot(shots[key]['z_accel_norm'])
 	plt.show()

 	#feat2.append(feature2(shots[key]))




def feature2(shot):
	z = [z for x,y,z in shot['accel_norm']]
	idx = z.index(min(z))

	v = []
	for i in range(idx-25,idx+1):
		x,y,z = shot['accel_norm'][i]
		v.append((x,y))

	x_avg= sum([x for x,y in v])
	y_avg = sum([y for x,y in v])

	dot_sum = []
	for x,y in v:
		dot_sum.append(abs((x*x_avg)+(y*y_avg)))

	return min(dot_sum)



def feature4(shot):
	z = [z for x,y,z in shot['accel_norm']]
	idx = z.index(min(z))

	v = []
	for i in range(idx-25,idx+1):
		x,y,z = shot['accel_norm'][i]
		v.append((x,y,z))

	x_avg= sum([x for x,y,z in v])
	y_avg = sum([y for x,y,z in v])
	z_avg = sum([z for x,y,z in v])

	dot_sum = []
	for x,y,z in v:
		dot_sum.append(abs((x*x_avg)+(y*y_avg)+(z*z_avg)))

	return min(dot_sum)

def feature5(shot):
	z = [z for x,y,z in shot['accel_norm']]
	idx = z.index(min(z))

	v = []
	for i in range(idx-25,idx+1):
		x,y,z = shot['accel_norm'][i]
		v.append((x,y,z))

	x_avg= sum([x for x,y,z in v])
	y_avg = sum([y for x,y,z in v])
	z_avg = sum([z for x,y,z in v])

	dot_sum = []
	for x,y,z in v:
		dot_sum.append((x*x_avg)+(y*y_avg)+(z*z_avg))

	return sum(dot_sum)


 	# feat5.append(feature5(shots[key]))
 	#feat3.append(feature3(shots[key]))
# 	max_accelx.append(max(shots[key]['accelx']))
# 	max_accely.append(max(shots[key]['accely']))
# 	max_accelz.append(max(shots[key]['accelz']))
# 	max_gyrox.append(max(shots[key]['gyrox']))

# X = []
# Y = []
# Z = []
# U = []
# V = []
# W = []

# i = 0
# for key in shots:
# 	x,y,z = quat_vec_at_release(shots[key])
# 	X.append(x)
# 	Y.append(y)
# 	Z.append(z)
# 	U.append(i/20)
# 	V.append(int(shots[key]['make']))
# 	W.append(0)
# 	i+= 1


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.quiver(U,V,W,X,Y,Z)
# ax.set_xlim([-1,10])
# ax.set_ylim([-1,3])
# ax.set_zlim([-1,2])
# plt.show()


# plt.scatter(feat2, make)
# plt.show()

# plt.scatter(feat5, make)
# plt.show()



# for key in shots:
# 	print(shots[key]['make'])
# 	print_shot(shots[key])

# def feature(shot):

# accel_base = rot_vectors(s['accel'], s['quat'])

# for item in accel_base:
# 	if magnitude(item) > 2:
# 		print(item)

# plt.plot(s['emg8'])
# plt.show()


# v1 = []
# for item in s['quat']:
# 	a,b,c,d = item
# 	v1.append((b,c,d))

# U = []
# V = []
# W = []
# for i in range(len(accel_base)):
# 	U.append(i/20)
# 	V.append(0)
# 	W.append(0)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # X = [x_ for x_,y_,z_ in s['accel']]
# # Y = [y_ for x_,y_,z_ in s['accel']]
# # Z = [z_ for x_,y_,z_ in s['accel']]
# X = []
# Y = []
# Z = []

# for item in accel_base:
# 	if magnitude(item) > 2:
# 		x,y,z = item
# 		X.append(x)
# 		Y.append(y)
# 		Z.append(z)
# 	else:
# 		X.append(0)
# 		Y.append(0)
# 		Z.append(0)

# ax.quiver(U,V,W,X,Y,Z)
# ax.set_xlim([-1,10])
# ax.set_ylim([-1,2])
# ax.set_zlim([-1,3])
# plt.show()


# for item1, item2 in zip(accel_base, s['accel']):
# 	print(item1)
# 	print(item2)
# 	print()

#print(rot_matrix())

# print_shot(s)


# for key in shots:
# 	print(key)
# 	print_shot(shots[key])

# plt.plot(shots[1]['accel'])




