class CA1Dview(object):

    """
    Provides a UI for a CA1D object.
    Can be replaced by any class providing the same methods.
    """

    def __init__(self, off_color, on_color):

        """
        These cryptic attributes use ANSI terminal codes to print a space
        in either the off colour or the on colour, resetting the colour at the end.
        """

        self.off_color = "\033[0;" + off_color + "m \033[0m"
        self.on_color = "\033[0;" + on_color + "m \033[0m"

    def print_ca(self, ca):

        """
        Before the first iteration calls show_properties.
        Then prints the iteration number as a row heading.
        Finally iterates the cells, printing either the off_color
        or on_color string.
        """

        if ca.iteration == 0:
            self.show_properties(ca)

        print(str(ca.iteration).ljust(2) + " ", end = '')

        for c in ca.cells:

            if c == "0":
                print(self.off_color, end = '')
            else:
                print(self.on_color, end = '')

        print("")

    def show_properties(self, ca):

        """
        Short utility function to output the cellular automaton's attributes
        """

        print("cell_count:   " + str(ca.cell_count))
        print("init_pattern: " + ca.init_pattern)
        print("rule:         " + str(ca.rule))
        print("iterations:   " + str(ca.iterations) + "\n")