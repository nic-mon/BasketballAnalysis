import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math



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

def parse_cont_data(data):
	'''parse data from continous subject with 2 myos'''
	d = {}
	d['myo1'] = []
	d['myo2'] = []
	
	for line in data[2:]:
		d['myo1'].append(float(line.split()[0]))
		d['myo2'].append(float(line.split()[1]))

	while d['myo1'][-1] == 0 or d['myo2'][-1] == 0:
		d['myo1'].pop()
		d['myo2'].pop()

	return d

def read_cont_subject(i):
	'''read data from continous subject with 2 myos'''
	accelx = parse_cont_data(read_file('DataCollectionP2/Subject{}/accelx.txt'.format(i)))
	accely = parse_cont_data(read_file('DataCollectionP2/Subject{}/accely.txt'.format(i)))
	accelz = parse_cont_data(read_file('DataCollectionP2/Subject{}/accelz.txt'.format(i)))
	faccelx = parse_cont_data(read_file('DataCollectionP2/Subject{}/faccelx.txt'.format(i)))
	faccely = parse_cont_data(read_file('DataCollectionP2/Subject{}/faccely.txt'.format(i)))
	faccelz = parse_cont_data(read_file('DataCollectionP2/Subject{}/faccelz.txt'.format(i)))

	# gyrox = parse_cont_data(read_file('DataCollectionP2/Subject{}/gyrox.txt'.format(i)))
	# gyroy = parse_cont_data(read_file('DataCollectionP2/Subject{}/gyroy.txt'.format(i)))
	# gyroz = parse_cont_data(read_file('DataCollectionP2/Subject{}/gyroz.txt'.format(i)))
	# fgyrox = parse_cont_data(read_file('DataCollectionP2/Subject{}/fgyrox.txt'.format(i)))
	# fgyroy = parse_cont_data(read_file('DataCollectionP2/Subject{}/fgyroy.txt'.format(i)))
	# fgyroz = parse_cont_data(read_file('DataCollectionP2/Subject{}/fgyroz.txt'.format(i)))

	#imutimes = parse_data(read_file('DataCollectionP2/Subject{}/imutimes.txt'.format(i)))
	#subjectinfo = read_file('DataCollectionP2/Subject{}/subjectinfo.txt'.format(i))

	quat1 = parse_cont_data(read_file('DataCollectionP2/Subject{}/quat1.txt'.format(i)))
	quat2 = parse_cont_data(read_file('DataCollectionP2/Subject{}/quat2.txt'.format(i)))
	quat3 = parse_cont_data(read_file('DataCollectionP2/Subject{}/quat3.txt'.format(i)))
	quat4 = parse_cont_data(read_file('DataCollectionP2/Subject{}/quat4.txt'.format(i)))
	# rot1 = parse_data(read_file('DataCollectionP2/Subject{}/rot1.txt'.format(i)))
	# rot2 = parse_data(read_file('DataCollectionP2/Subject{}/rot2.txt'.format(i)))
	# rot3 = parse_data(read_file('DataCollectionP2/Subject{}/rot3.txt'.format(i)))


	d = {}
	#d['subjectinfo'] = subjectinfo

	d['faccelx'] = faccelx
	d['faccely'] = faccely
	d['faccelz'] = faccelz
	d['accelx'] = accelx
	d['accely'] = accely
	d['accelz'] = accelz
	d['quat1'] = quat1
	d['quat2'] = quat2
	d['quat3'] = quat3
	d['quat4'] = quat4

	d['myo1'] = {}
	d['myo2'] = {}
	d['myo1']['quat'] = list(zip(d['quat1']['myo1'], d['quat2']['myo1'], d['quat3']['myo1'], d['quat4']['myo1']))
	d['myo2']['quat'] = list(zip(d['quat1']['myo2'], d['quat2']['myo2'], d['quat3']['myo2'], d['quat4']['myo2']))
	d['myo1']['faccel'] = list(zip(d['faccelx']['myo1'], d['faccely']['myo1'], d['faccelz']['myo1']))
	d['myo2']['faccel'] = list(zip(d['faccelx']['myo2'], d['faccely']['myo2'], d['faccelz']['myo2']))

			# d['shot{}'.format(j+1)]['emgtimes'] = emgtimes["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['gyrox'] = gyrox["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['gyroy'] = gyroy["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['gyroz'] = gyroz["shot{}".format(j+1)]
			# d['shot{}'.format(j+1)]['imutimes'] = imutimes["shot{}".format(j+1)]

	return d

def rot_vectors(vectors, quats):
	'''rotates vectors by quaternions, giving fixed axis vectors'''
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


def rot_x_unit3(quat):
	'''rotates a unit x vector by given quaternion'''
	a,b,c,d = quat
	vx = 1 - 2*c*c - 2*d*d
	vy = 2*(b*c - a*d)
	vz = 2*(b*d + a*c)

	return (vx, vy, vz)

def rot_x_unit4(quat):
	a,b,c,d = quat
	vx = 1 - 2*c*c - 2*d*d
	vy = 2*(b*c - a*d)
	vz = 2*(b*d + a*c)

	return (-vx, -vy, -vz)

import math

def quat_to_euler(quat):
	'''converts a quaternion to a euler angle'''
	a,b,c,d = quat
	yaw = math.atan(2*(a*b+c*d) / (a*a - b*b - c*c + d*d))
	if( -1 > 2*(b*d - a*c) or 1< 2*(b*d - a*c)):
		print('oops', 2*(b*d - a*c))
	pitch = -1 * math.asin(max(min(2*(b*d - a*c),1),-1))
	roll = math.atan(2*(a*d+b*c) / (a*a + b*b - c*c - d*d))
	return (yaw, pitch, roll)

def magnitude(a):
	'''computes magnitute of given vector'''
	x,y,z = a
	return math.sqrt(x**2 + y**2 + z**2)

def dot_prod(v1, v2):
	'''computes dot product of two 3D vectors'''
	a,b,c = v1
	d,e,f = v2
	return (a*d)+(b*e)+(c*f)

def x_dot(subj):
	'''computes dot product of unit direction vectors from the two myos'''
	myo1_q = list(zip(subj['quat1']['myo1'], subj['quat2']['myo1'], subj['quat3']['myo1'], subj['quat4']['myo1']))
	myo2_q = list(zip(subj['quat1']['myo2'], subj['quat2']['myo2'], subj['quat3']['myo2'], subj['quat4']['myo2']))
	
	return [dot_prod(rot_x_unit3(q1),rot_x_unit3(q2)) for q1, q2 in zip(myo1_q, myo2_q)]

def xy_dot_prod(v1, v2):
	'''computes the dot product of only the X and Y components of 2 vectors'''
	a,b,c = v1
	d,e,f = v2
	return (a*d) + (b*e)


def xy_dot(subj):
	myo1_q = list(zip(subj['quat1']['myo1'], subj['quat2']['myo1'], subj['quat3']['myo1'], subj['quat4']['myo1']))
	myo2_q = list(zip(subj['quat1']['myo2'], subj['quat2']['myo2'], subj['quat3']['myo2'], subj['quat4']['myo2']))
	
	return [xy_dot_prod(rot_x_unit3(q1),rot_x_unit3(q2)) for q1, q2 in zip(myo1_q, myo2_q)]

def rad_to_deg(rad):
	return rad*180 / math.pi

def angle_from_vec(vec):
	x,y,z = vec
	xy = math.sqrt(x*x + y*y)
	return 90 - rad_to_deg(math.atan(z / xy))

d = {}
d['subject1'] = read_cont_subject(7)
d['subject2'] = read_cont_subject(9)

# plt.plot(d['subject3']['faccelx']['myo2'])
# plt.show()

# plt.plot(d['subject5']['faccelx']['myo2'])
# plt.show()

s1_i = [892, 1306, 1737, 2174, 2605, 3035, 3479, 3998, 4673, 5075, 5496, 6049, 6512]
s2_i = [885, 1513, 2013, 2323, 2644, 2883, 3154, 3526, 3862]

my1z= [rot_x_unit3(q)[2] for q in d['subject1']['myo1']['quat']]

for i in s1_i:
	j = np.argmin(my1z[i-25:i+10])
	print(j)
	#k = np.argmin(rot_x_unit4(d['subject1']['myo2']['quat'][i-25:i+25][2])
	print((i/50)-3.5, xy_dot_prod(rot_x_unit3(d['subject1']['myo2']['quat'][i-25+j]),rot_x_unit3(d['subject1']['myo1']['quat'][i-25+j])))


# i = 0
# for z in d['subject2']['faccelz']['myo2']:
# 	i += 1
# 	if z < -3:
#  		print(i, z, ed[i])

# i = 0
# for z in d['subject2']['faccelz']['myo2']:
# 	i += 1
# 	if z < -3:
# 		print(i, (i/50)-3.5, z)

x_dot = x_dot(d['subject2'])

import numpy as np
for i in s2_i:
	j = np.argmax(x_dot[i-25:i+25])
	print(j)
	print((i/50)-7, angle_from_vec(rot_x_unit4(d['subject2']['myo2']['quat'][i])))




# plt.plot(d['subject1']['faccelz']['myo2'])
# plt.show()

# plt.plot(d['subject2']['faccelx']['myo2'])
# plt.show()
