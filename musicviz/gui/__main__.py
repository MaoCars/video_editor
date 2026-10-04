import multiprocessing as mp

from .app import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
