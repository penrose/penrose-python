'''

Copyright (C) 2019 Vanessa Sochat.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

from penrose.logger import bot
import errno
import shutil
import json
import tempfile
import sys
import os
import re

################################################################################
# Directories
################################################################################

def find_subdirectories(basepath):
    '''
    Return directories (and sub) starting from a base
    '''
    directories = []
    for root, dirnames, filenames in os.walk(basepath):
        new_directories = [d for d in dirnames if d not in directories]
        directories = directories + new_directories
    return directories

    
def find_directories(root):
    '''
    Return directories at one level specified by user (not recursive).
    We don't include hidden directories.
    '''
    directories = []
    for item in os.listdir(root):
        if not re.match("^[.]",item):
            if os.path.isdir(os.path.join(root, item)):
                directories.append(os.path.abspath(os.path.join(root, item)))
    return directories

 
def copy_directory(src, dest, force=False):
    ''' Copy an entire directory recursively
    '''
    if os.path.exists(dest) and force is True:
        shutil.rmtree(dest)
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            bot.error('Directory not copied. Error: %s' % e)
            sys.exit(1)


def mkdir_p(path):
    '''mkdir_p attempts to get the same functionality as mkdir -p
    :param path: the path to create.
    '''
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            bot.error("Error creating path %s, exiting." % path)
            sys.exit(1)


################################################################################
# File Utils
################################################################################

# Json

def get_installdir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_json(filename, mode='r'):
    with open(filename, mode) as filey:
        data = json.load(filey)
    return data

def write_json(json_obj, filename, mode='w'):
    with open(filename,mode) as filey:
        filey.write(json.dumps(json_obj, sort_keys=True,indent=4, separators=(',', ': ')))
    return filename

# Text file

def read_file(filename,mode='r'):
    with open(filename,mode) as filey:
        data = filey.read()
    return data

def write_file(filename,content,mode='w'):
    with open(filename,mode) as filey:
        filey.writelines(content)
    return filename

def get_post_fields(request):
    '''parse through a request, and return fields from post in a dictionary

       Parameters
       ==========
       request: a request object
    '''
    fields = dict()
    for field,value in request.form.items():
        fields[field] = value
    return fields



################################################################################
# environment / options
################################################################################

def convert2boolean(arg):
    '''convert2boolean is used for environmental variables
    that must be returned as boolean'''
    if not isinstance(arg, bool):
        return arg.lower() in ("yes", "true", "t", "1", "y")
    return arg


def make_lower(arg):
    '''helper function to make lowercase or return original'''
    if arg is not None:
        if isinstance(arg, str):
            arg = arg.tolower()
    return arg


def getenv(variable_key, default=None, required=False, silent=True):
    '''getenv will attempt to get an environment variable. If the variable
    is not found, None is returned.

    Parameters
    ==========
    variable_key: the variable name
    required: exit with error if not found
    silent: Do not print debugging information for variable
    
    '''
    variable = os.environ.get(variable_key, default)
    if variable is None and required:
        bot.error("Cannot find environment variable %s, exiting." % variable_key)
        sys.exit(1)

    if not silent:
        if variable is not None:
            bot.verbose2("%s found as %s" %(variable_key,variable))
        else:
            bot.verbose2("%s not defined (None)" %variable_key)

    return variable
