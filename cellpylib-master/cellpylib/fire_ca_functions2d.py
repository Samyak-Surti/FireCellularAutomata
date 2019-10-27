from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy as np
import math

def plot2d_animate(ca, title=''):
    cmap = plt.get_cmap('Reds')
    fig = plt.figure()
    plt.title(title)
    im = plt.imshow(ca[0], animated=True, cmap=cmap)
    i = {'index': 0}
    def updatefig(*args):
        i['index'] += 1
        if i['index'] == len(ca):
            i['index'] = 0
        im.set_array(ca[i['index']])
        return im,
    ani = animation.FuncAnimation(fig, updatefig, interval=50, repeat=False, blit=True)
    plt.show()

def evolve2d(cellular_automaton, timesteps, apply_rule, r=1, neighbourhood='Moore'):
    # """
    #
    # :param cellular_automaton:
    # :param timesteps: the number of time steps in this evolution; note that this value refers to the total number of
    #                   time steps in this cellular automaton evolution, which includes the initial condition
    # :param apply_rule: a function representing the rule to be applied to each cell during the evolution; this function
    #                    will be given three arguments, in the following order: the neighbourhood, which is a numpy
    #                    2D array of dimensions 2r+1 x 2r+1, representing the neighbourhood of the cell (if the
    #                    'von Neumann' neighbourhood is specified, the array will be a masked array); the cell identity,
    #                    which is a tuple representing the row and column indices of the cell in the cellular automaton
    #                    matrix, as (row, col); the time step, which is a scalar representing the time step in the
    #                    evolution
    # :param r: the neighbourhood radius; the neighbourhood dimensions will be 2r+1 x 2r+1
    # :param neighbourhood: the neighbourhood type; valid values are 'Moore' or 'von Neumann'
    # :return:
    # """
    stacks, rows, cols = cellular_automaton.shape
    array = np.zeros((timesteps, stacks, rows, cols), dtype=cellular_automaton.dtype)
    #print (array.shape)
    #print (array[0].shape)
    array[0] = cellular_automaton

    von_neumann_mask = np.zeros((2*r + 1, 2*r + 1), dtype=bool)
    for i in range(len(von_neumann_mask)):
        mask_size = np.absolute(r - i)
        von_neumann_mask[i][:mask_size] = 1
        if mask_size != 0:
            von_neumann_mask[i][-mask_size:] = 1

    def get_neighbourhood(cell_layer, row, col):
        row_indices = range(row - r, row + r + 1)
        row_indices = [i - cell_layer.shape[0] if i > (cell_layer.shape[0] - 1) else i for i in row_indices]
        col_indices = range(col - r, col + r + 1)
        col_indices = [i - cell_layer.shape[1] if i > (cell_layer.shape[1] - 1) else i for i in col_indices]
        n = cell_layer[np.ix_(row_indices, col_indices)]
        if neighbourhood == 'Moore':
            return n
        elif neighbourhood == 'von Neumann':
            return np.ma.masked_array(n, von_neumann_mask)
        else:
            raise Exception("unknown neighbourhood type: %s" % neighbourhood)

    for t in range(1, timesteps):
    	cell_layer = array[t - 1]
    	for row in range(cell_layer[0][0].size):
    		for col in range(cell_layer[0][0].size):
    			#print (cell_layer.shape)
    			n = get_fire_neighborhood(cell_layer, row, col)
    			n = apply_rule(n, (row, col), t)
    			for i in range(3):
    				array[t][i][row][col] = n[i][1][1]


def init_random_fire_map2d(rows, cols, k = 1, dtype = np.float_, spread='random'):
	"""
	Returns a randomly initialized matrix with floating point numbers rounded to three digits between 0 and 1.
	:param rows: the number of rows in the matrix
	:param cols: the number of columns in the atrix
	:param k: the cap for the probability of cell settingo n fire
	:param dtype: the data type
	:return: a tensor with shape (1, rows, cols), randomly initialized with unformly distributed numbers from 0 to 1.
	"""
	if spread == 'random':
		fuel_map = np.random.normal(0.6, 0.2, size = (rows, cols)).astype(dtype)
		fuel_map = np.array([fuel_map])
	if spread == 'uniform':
		val = np.random.normal(0.5, 0.1)
		print (val)
		fuel_map = np.full((rows, cols), val)
		fuel_map = np.array([fuel_map])
	state_map = np.zeros((rows,cols), dtype=dtype)
	state_map = np.array([state_map])
	spread_prob_map = np.zeros((rows, cols), dtype = dtype)
	spread_prob_map = np.array([spread_prob_map])
	main_map = np.reshape(np.vstack((fuel_map, state_map, spread_prob_map)),(3,rows,cols))
	for i in range(1):
		main_map[1, random.randint(0,rows-1), random.randint(0,cols-1)] = 1
	for row in range(rows):
		for col in range(cols):
			main_map[0, row, col] = round(main_map[0, row, col], 5)
			if main_map[0, row, col] < 0:
				main_map[0, row, col] = 0
	return main_map
	
#Finds the Moore neighborhood of r = 1 for any given cell.
def get_fire_neighborhood(map, center_r_val, center_c_val):
	neighborhood = np.zeros((3,3,3))
	if (center_r_val == 0 and center_c_val == 0):
		for a in range(3):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i+1][j+1] = map[a][center_r_val + i][center_c_val + j]
		#print neighborhood
		return neighborhood
	if (center_r_val == 0 and center_c_val == map[0][0].size - 1):
		for a in range(3):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i+1][j] = map[a][center_r_val + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_r_val == map[0][0].size - 1 and center_c_val == 0):
		for a in range(3):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i][j+1] = map[a][(center_r_val - 1) + i][center_c_val + j]
		return neighborhood
	if (center_r_val == map[0][0].size - 1 and center_c_val == map[0][0].size - 1):
		for a in range(3):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_r_val == 0):
		for a in range(3):
			for i in range(2):
				for j in range(3):
					neighborhood[a][i+1][j] = map[a][center_r_val + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_c_val == 0):
		for a in range(3):
			for i in range(3):
				for j in range(2):
					neighborhood[a][i][j+1] = map[a][(center_r_val - 1) + i][center_c_val + j]
		return neighborhood
	if (center_r_val == map[0][0].size - 1):
		for a in range(3):
			for i in range(2):
				for j in range(3):
					neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_c_val == map[0][0].size - 1):
		for a in range(3):
			for i in range(3):
				for j in range(2):
					neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
		return neighborhood
	for a in range(3):
		for i in range(3):
			for j in range(3):
				neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
	return neighborhood

def fire_spread_rule(neighborhood, c, t): 
	#print (t)
	row_val, col_val = c
	fire_load_center = neighborhood[0][1][1]
	state_center = neighborhood[1][1][1]
	for i in range(3):
		for j in range(3):
			if (i == 0 and j == 0) or (i == 2 and j == 0) or (i == 0 and j == 2) or (i == 2 and j == 2):
				#print (neighborhood[2][1][1])
				neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4)
			else:
				neighborhood[2][1][1] += (neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)
			#neighborhood[2][1][1] = neighborhood[2][1][1] * fire_load_center
	#neighborhood[2][1][1] = neighborhood[2][1][1]/8
	if neighborhood[1][1][1] == 1:
		neighborhood[0][1][1] -= 0.005
		if (neighborhood[0][1][1] <= 0):
			neighborhood[0][1][1] = 0
			neighborhood[1][1][1] = 0
	if (neighborhood[1][1][1] == 0 and neighborhood[2][1][1] > 4.0):
		neighborhood[1][1][1] = 1
	return neighborhood
	
