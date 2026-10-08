"""Punto de entrada de la aplicación empaquetada (ventana)."""
import multiprocessing as mp
import sys

if __name__ == "__main__":
    mp.freeze_support()
    from musicviz.gui.app import main

    main(sys.argv[1:])
