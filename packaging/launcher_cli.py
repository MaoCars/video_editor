"""Punto de entrada de la línea de comandos empaquetada."""
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    from musicviz.cli import main

    main()
