class CA1D(object):

    def __init__(self, cell_count, init_pattern, rule, iterations, on_change):

        """
        Creates attributes with values from arguments or defaults.
        Set initial state of cells from init_pattern
        and then calls the on_change function to let whatever UI
        has been plugged in to update the output.
        """

        self.cell_count = cell_count
        self.init_pattern = init_pattern
        self.rule = rule
        self.iterations = iterations
        self.on_change = on_change
        self.iteration = 0
        self.cells = []
        self.__next_state = []
        self.rule_binary = format(self.rule, '08b')

        # set cells from init pattern
        for c in self.init_pattern:

            if c == "0":
                self.cells.append("0")
            elif c == "1":
                self.cells.append("1")

            self.__next_state.append("0")

        # call on_change to let UI know CA has been created
        self.on_change(self)

    def start(self):

        """
        Loop for specified number of iterations,
        calculating next state and updating UI
        """

        neighbourhood = ""

        for i in range(0, self.iterations):

            self.iteration += 1

            self.__calculate_next_state()

            self.on_change(self)

    def __calculate_next_state(self):

        """
        For each cell, calculate that cells next state depending on the current rule.
        Then copy the next state to the current state
        """

        for c in range(0, self.cell_count - 1):

            if c == 0:
                # roll beginning round to end
                prev_index = self.cell_count - 1
            else:
                prev_index = c - 1

            if c == (self.cell_count - 1):
                # roll end round to beginning
                next_index = 0
            else:
                next_index = c + 1

            neighbourhood = self.cells[prev_index] + self.cells[c] + self.cells[next_index]

            if neighbourhood == "111":
                self.__next_state[c] = self.rule_binary[0]
            elif neighbourhood == "110":
                self.__next_state[c] = self.rule_binary[1]
            elif neighbourhood == "101":
                self.__next_state[c] = self.rule_binary[2]
            elif neighbourhood == "100":
                self.__next_state[c] = self.rule_binary[3]
            elif neighbourhood == "011":
                self.__next_state[c] = self.rule_binary[4]
            elif neighbourhood == "010":
                self.__next_state[c] = self.rule_binary[5]
            elif neighbourhood == "001":
                self.__next_state[c] = self.rule_binary[6]
            elif neighbourhood == "000":
                self.__next_state[c] = self.rule_binary[7]

        for c in range(0, self.cell_count):
            self.cells[c] = self.__next_state[c]