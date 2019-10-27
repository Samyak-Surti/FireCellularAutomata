from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy as np
import math


def plot2d(ca, timestep=None, title=''):
    cmap = plt.get_cmap('Reds')
    plt.title(title)
    if timestep is not None:
        data = ca[timestep]
    else:
        data = ca[-1]
    plt.imshow(data, interpolation='none', cmap=cmap)
    plt.show()


def plot2d_slice(ca, slice=None, title=''):
    cmap = plt.get_cmap('Greys')
    plt.title(title)
    if slice is not None:
        data = ca[:, slice]
    else:
        data = ca[:, len(ca[0])//2]
    plt.imshow(data, interpolation='none', cmap=cmap)
    plt.show()


def plot2d_spacetime(ca, alpha=None, title=''):
    fig = plt.figure(figsize=(10, 7))
    plt.title(title)
    ax = fig.gca(projection='3d')
    ca = ca[::-1]
    xs = np.arange(ca.shape[2])[None, None, :]
    ys = np.arange(ca.shape[1])[None, :, None]
    zs = np.arange(ca.shape[0])[:, None, None]
    xs, ys, zs = np.broadcast_arrays(xs, ys, zs)
    masked_data = np.ma.masked_where(ca == 0, ca)
    ax.scatter(xs.ravel(),
               ys.ravel(),
               zs.ravel(),
               c=masked_data, cmap='cool', marker='s', depthshade=False, alpha=alpha, edgecolors='#0F0F0F')
    plt.show()

#Animates the cellular automata for each timestep
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

#Changes the configuration of the cellular automata from the previous timestep by considering the neighborhoods of each cell and the rule being applied
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
    			#print("evaluating row = ",row, "col = ",col)
    			n = get_fire_neighborhood(cell_layer, row, col)
    			n = apply_rule(n, (row, col), t)
    			#print(n[2][1][1])
    			for i in range(6):
    				array[t][i][row][col] = n[i][1][1]

    """
    for t in range(1, timesteps):
        cell_layer = array[t - 1]
        for row, cell_row in enumerate(cell_layer):
            for col, cell in enumerate(cell_row):
                n = get_neighbourhood(cell_layer, row, col)
                print (n.shape)
                array[t][row][col] = apply_rule(n, (row, col), t)
    """
    return array


def init_simple2d(rows, cols, val=1, dtype=np.int):
    """
    Returns a matrix initialized with zeroes, with its center value set to the specified value, or 1 by default.
    :param rows: the number of rows in the matrix
    :param cols: the number of columns in the matrix 
    :param val: the value to be used in the center of the matrix (1, by default)
    :param dtype: the data type
    :return: a tensor with shape (1, rows, cols), with the center value initialized to the specified value, or 1 by default 
    """
    x = np.zeros((rows, cols), dtype=dtype)
    x[x.shape[0]//2][x.shape[1]//2] = val
    return np.array([x])


def init_random2d(rows, cols, k=2, dtype=np.int):
    """
    Returns a randomly initialized matrix with values consisting of numbers in {0,...,k - 1}, where k = 2 by default.
    If dtype is not an integer type, then values will be uniformly distributed over the half-open interval [0, k - 1).
    :param rows: the number of rows in the matrix
    :param cols: the number of columns in the matrix 
    :param k: the number of states in the cellular automaton (2, by default)
    :param dtype: the data type
    :return: a tensor with shape (1, rows, cols), randomly initialized with numbers in {0,...,k - 1}
    """
    if np.issubdtype(dtype, np.integer):
        rand_nums = np.random.randint(k, size=(rows, cols), dtype=dtype)
    else:
        rand_nums = np.random.uniform(0, k - 1, size=(rows, cols)).astype(dtype)
    return np.array([rand_nums])

#Initializes all the layers of the cellular automata
def init_random_fire_map2d(rows, cols, k = 1, dtype = np.float_, spread='random'):
	"""
	Returns a randomly initialized matrix with floating point numbers rounded to three digits between 0 and 1.
	:param rows: the number of rows in the matrix
	:param cols: the number of columns in the atrix
	:param k: the cap for the probability of cell settingo n fire
	:param dtype: the data type
	:return: a tensor with shape (6, rows, cols), with all the layers of the cellular automata such as the fuel load, states, state change factor, 
	wind vector components, and topographic feature components.
	"""
	if spread == 'random':
		fuel_map = np.random.normal(0.6, 0.2, size = (rows, cols)).astype(dtype)
		fuel_map = np.array([fuel_map])
	if spread == 'uniform':
		val = np.random.normal(0.5, 0.1)
		fuel_map = np.full((rows, cols), val)
		fuel_map = np.array([fuel_map])
	state_map = np.zeros((rows,cols), dtype=dtype)
	state_map = np.array([state_map])
	spread_prob_map = np.zeros((rows, cols), dtype = dtype)
	spread_prob_map = np.array([spread_prob_map])
	wind_x_map = np.full((rows, cols), 0)
	wind_x_map = np.array([wind_x_map])
	wind_y_map = np.full((rows, cols), 0)
	wind_y_map = np.array([wind_y_map])
	topo_map = np.zeros((rows,cols))
	
	"""
	for r in range(rows):
		for c in range(cols):
			topo_map[r][c] = r + c
	"""
			
	topo_map = np.array([topo_map])
	main_map = np.reshape(np.vstack((fuel_map, state_map, spread_prob_map, wind_x_map, wind_y_map, topo_map)),(6,rows,cols))
	for i in range(1):
		#main_map[1, random.randint(0,rows-1), random.randint(0,cols-1)] = 1
		main_map[1, (rows-1)/2, (cols-1)/2] = 1
	for row in range(rows):
		for col in range(cols):
			main_map[0, row, col] = round(main_map[0, row, col], 5)
			if main_map[0, row, col] < 0:
				main_map[0, row, col] = 0
	return main_map
	
#NOTE: c is the row-column tuple that defines center of neighborhood and t is the timestep number
def game_of_life_rule(neighbourhood, c, t):
    center_cell = neighbourhood[1][1]
    total = np.sum(neighbourhood)
    if center_cell == 1:
        if total - 1 < 2:
            return 0  # Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        if total - 1 == 2 or total - 1 == 3:
            return 1  # Any live cell with two or three live neighbours lives on to the next generation.
        if total - 1 > 3:
            return 0  # Any live cell with more than three live neighbours dies, as if by overpopulation.
    else:
        if total == 3:
            return 1  # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        else:
            return 0

#Finds the Moore neighborhood of r = 1 for any given cell.
def get_fire_neighborhood(map, center_r_val, center_c_val):
	num_layers = 6
	neighborhood = np.zeros((num_layers,3,3))
	if (center_r_val == 0 and center_c_val == 0):
		for a in range(num_layers):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i+1][j+1] = map[a][center_r_val + i][center_c_val + j]
		#print neighborhood
		return neighborhood
	if (center_r_val == 0 and center_c_val == map[0][0].size - 1):
		for a in range(num_layers):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i+1][j] = map[a][center_r_val + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_r_val == map[0][0].size - 1 and center_c_val == 0):
		for a in range(num_layers):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i][j+1] = map[a][(center_r_val - 1) + i][center_c_val + j]
		return neighborhood
	if (center_r_val == map[0][0].size - 1 and center_c_val == map[0][0].size - 1):
		for a in range(num_layers):
			for i in range(2):
				for j in range(2):
					neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_r_val == 0):
		for a in range(num_layers):
			for i in range(2):
				for j in range(3):
					neighborhood[a][i+1][j] = map[a][center_r_val + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_c_val == 0):
		for a in range(num_layers):
			for i in range(3):
				for j in range(2):
					neighborhood[a][i][j+1] = map[a][(center_r_val - 1) + i][center_c_val + j]
		return neighborhood
	if (center_r_val == map[0][0].size - 1):
		for a in range(num_layers):
			for i in range(2):
				for j in range(3):
					neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
		return neighborhood
	if (center_c_val == map[0][0].size - 1):
		for a in range(num_layers):
			for i in range(3):
				for j in range(2):
					neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
		return neighborhood
	for a in range(num_layers):
		for i in range(3):
			for j in range(3):
				neighborhood[a][i][j] = map[a][(center_r_val - 1) + i][(center_c_val - 1) + j]
	return neighborhood

def fire_spread_rule(neighborhood, c, t): 
	#print (t)
	n_vec = []
	w_vec = []
	fire_load_center = neighborhood[0][1][1]
	state_center = neighborhood[1][1][1]
	h = 0
	b = 0
	slope = 0
	#print (w_vec)
	for i in range(3):
		for j in range(3):
			if (i == 0 and j == 0):
				n_vec = [1/math.sqrt(2),-1/math.sqrt(2)]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = math.sqrt(2)
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
			if (i == 2 and j == 0):
				n_vec = [1/math.sqrt(2), 1/math.sqrt(2)]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = math.sqrt(2)
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
			if (i == 0 and j == 2):
				n_vec = [-1/math.sqrt(2), -1/math.sqrt(2)]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = math.sqrt(2)
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
			if (i == 2 and j == 2):
				n_vec = [-1/math.sqrt(2), 1/math.sqrt(2)]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = math.sqrt(2)
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (math.pi/4) * (1 + (np.dot(n_vec,w_vec)))
			if (i == 0 and j == 1):
				n_vec = [0,-1]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = 1
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
			if (i == 1 and j == 0):
				n_vec = [1,0]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = 1
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
				#print("center state = ", state_center, "left = ", neighborhood[2][1][1])
			if (i == 2 and j == 1):
				n_vec = [0,1]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = 1
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
			if (i == 1 and j == 2):
				n_vec = [-1,0]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = 1
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
			if (i == 1 and j == 1):
				n_vec = [0,0]
				w_vec = [neighborhood[3][i][j],neighborhood[4][i][j]]
				h = neighborhood[5][1][1] - neighborhood[5][i][j]
				b = 1
				slope = float(h)/float(b)
				if (fire_load_center < 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
				if (fire_load_center >= 0.7):
					if h > 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + slope)
					if h < 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)) + math.pow(slope,2))
					if h == 0:
						neighborhood[2][1][1] += ((neighborhood[1][i][j] * neighborhood[0][i][j]) - (state_center * fire_load_center)) * (1 + (np.dot(n_vec,w_vec)))
	#print (neighborhood[2][:][:])
	if neighborhood[1][1][1] == 1:
		neighborhood[0][1][1] -= 0.005
		if (neighborhood[0][1][1] <= 0):
			neighborhood[0][1][1] = 0
			neighborhood[1][1][1] = 0
	if (neighborhood[1][1][1] == 0 and neighborhood[2][1][1] > 10.0 and neighborhood[0][1][1] != 0):
		neighborhood[1][1][1] = 1
	return neighborhood
	

