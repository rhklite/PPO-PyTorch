"""
File Description: custom print utilities for debug
Project: python
Author: Daniel Dworakowski, Richard Hu
Date: Nov-18-2019
"""

import inspect


class Colours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#
# Error information.


def lineInfo():
    callerframerecord = inspect.stack()[2]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1:]
    return '%s::%s:%d' % (file, info.function, info.lineno)
#
# Line information.


def getLineInfo(leveloffset=0):
    level = 2 + leveloffset
    callerframerecord = inspect.stack()[level]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = info.filename
    file = file[file.rfind('/') + 1:]
    return '%s: %d' % (file, info.lineno)
#
# Colours a string.


def colourString(msg, ctype):
    return ctype + msg + Colours.ENDC
#
# Print something in color.


def printColour(msg, ctype):
    print(colourString(msg, ctype))
#
# Print information.


def printInfo(*umsg):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in umsg:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.OKGREEN) + lst
    print(msg)
#
# Print error information.


def printFrame():
    print(lineInfo(), Colours.WARNING)
#
# Print an error.


def printError(*errstr):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in errstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.FAIL) + lst
    print(msg)
#
# Print a warning.


def printWarn(*warnstr):
    msg = '%s:  ' % (lineInfo())
    lst = ''
    for mstr in warnstr:
        lst += str(mstr) + ' '
    msg = colourString(msg, Colours.WARNING) + lst
    print(msg)
