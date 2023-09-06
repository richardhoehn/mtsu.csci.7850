#!/usr/bin/env python3

import sys

# Setup Config & app Parameters
if (len(sys.argv) < 4):
    raise Exception("You did not provide enough arguments")



print('Number of arguments:', len(sys.argv))
print('List of arguments:', str(sys.argv))
print('The first argument:', sys.argv[0])


