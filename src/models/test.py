import os, sys

from os.path import dirname, join, abspath
print(sys.path.insert(0, abspath(join(dirname(__file__), '..'))))

# from root_folder import file_name