import sys
sys.path.append('/Users/samyak/Documents/FireCellularAutomata/cellpylib-master')
import cellpylib as cpl

# Glider
cellular_automaton = cpl.init_simple2d(120, 120)
cellular_automaton[:, [28,29,30,30], [30,31,29,31]] = 1

# Blinker
cellular_automaton[:, [40,40,40], [15,16,17]] = 1

# Light Weight Space Ship (LWSS)
cellular_automaton[:, [18,18,19,20,21,21,21,21,20], [45,48,44,44,44,45,46,47,48]] = 1

# evolve the cellular automaton for 60 time steps
cellular_automaton = cpl.evolve2d(cellular_automaton, timesteps=120, neighbourhood='Moore',
                                  apply_rule=cpl.game_of_life_rule)

cpl.plot2d_animate(cellular_automaton)
