import sys
sys.path.append('/home/samyaknn/Documents/FireCellularAutomata/cellpylib-master')
import numpy as np
import cellpylib as cpl

ca_rows = 50
ca_cols = 50
num_timesteps = 120
num_layers = 6
val_spread = 'uniform'
cellular_automaton = cpl.init_random_fire_map2d(ca_rows, ca_cols,spread=val_spread) 
val_string = ""
state_string = ""
neighborhood_string = ""
fire_cells = np.array([])
topo_cells = np.array([])



	
cellular_automaton = cpl.evolve2d(cellular_automaton, timesteps=num_timesteps, neighbourhood='Moore', apply_rule=cpl.fire_spread_rule)
for i in range(num_timesteps):
	#print (cellular_automaton[i][2][:][:])
	fire_cells = np.append(fire_cells, (cellular_automaton[i][1][:][:]))
	#topo_cells = np.append(topo_cells, (cellular_automaton[i][5][:][:]))
fire_cells = np.reshape(fire_cells, (num_timesteps, ca_rows, ca_cols))

#topo_cells = np.reshape(topo_cells, (num_timesteps, ca_rows, ca_cols))
cpl.plot2d_animate(fire_cells)
#cpl.plot2d(fire_cells, title = 'Uniform Fire Load (Wind and Topographic Features Considered)')
#cpl.plot2d_animate(topo_cells)

"""
#Prints out fuel load probability values
for row in range(ca_rows):
	for col in range(ca_cols):
		#cellular_automaton[0, row, col] = round(cellular_automaton[0, row, col], 5)
		val_string += str(cellular_automaton[:,0, row, col]) + " "
	print (val_string + "\n")
	val_string = ""

#Prints out state set values
for r in range(ca_rows):
	for c in range(ca_cols):
		if cellular_automaton[1, r, c] == 1.0:
			fire_cells.append((r,c))
		state_string += str(cellular_automaton[1, r, c]) + " "
	print (state_string + "\n")
	state_string = ""
	

#Retrieves and prints out neighborhood for any arbitrary center cell.
neighborhood = get_fire_neighborhood(cellular_automaton, 2, 2)
for i in range(3):
	for j in range(3):
		neighborhood_string += str(neighborhood[0][i][j]) + " "
	print (neighborhood_string + "\n")	
	neighborhood_string = "" 

for a in range(3):
	for b in range(3):
		neighborhood_string += str(neighborhood[1][a][b]) + " "
	print (neighborhood_string + "\n")	
	neighborhood_string = "" 
"""


	

	



