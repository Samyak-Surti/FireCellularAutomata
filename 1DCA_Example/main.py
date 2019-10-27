import ca1d
import ca1dview

def main():

    """
    Demonstration of the CA1D and CA1Dview classes.
    Creates a CA1Dview and then a CA1D.
    The CA1Dview's print_ca method is passed to CA1D to provide a
    "plug-in" front end.
    """

    print("-------------------------\n| code-in-python.com    |\n| Cellular Automaton 1D |\n-------------------------\n")

    # Valid colours
    # 40 black, 41 red, 42 green, 43 yellow, 44 blue, 45 purple, 46 cyan, 47 white
    cav = ca1dview.CA1Dview("47", "40")

    ca = ca1d.CA1D(cell_count = 66,
                   init_pattern = "000000000000000000000000000000000100000000000000000000000000000000",
                   rule = 90,
                   iterations = 30,
                   on_change = cav.print_ca)

    ca.start()

#-----------------------------------------------------

main()