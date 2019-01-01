#!/usr/bin/env python

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

from glob import glob
import argparse
import sys
import os


def get_parser():

    parser = argparse.ArgumentParser(
    description="penrose-lib: interact with the Penrose Library")

    subparsers = parser.add_subparsers(help='Penrose Library actions',
                                       title='actions',
                                       description='actions for penrose-lib',
                                       dest="command")

    parser.add_argument("--version", dest='version', 
                        help="Get the version of penrose python installed", 
                        type=str, default=None)

    # List
    listy = subparsers.add_parser("list", 
                                   help="List components in the Penrose Library")


    # Install
    install = subparsers.add_parser("install", 
                                     help="install a component from the Penrose Library")

    install.add_argument('src', nargs=1, help='source url or folder of experiment')
    install.add_argument("--folder", dest='folder', 
                          help="empty folder to install experiment, defaults to pwd", 
                          type=str, default=None)

    install.add_argument('--force', '-f',dest="force",
                         help="force installation into non empty directory",
                         default=False, action='store_true')

    return parser


def get_subparsers(parser):
    '''get a dictionary of subparsers to help with printing help
    '''
    actions = [action for action in parser._actions 
               if isinstance(action, argparse._SubParsersAction)]

    subparsers = dict()
    for action in actions:
        # get all subparsers and print help
        for choice, subparser in action.choices.items():
            subparsers[choice] = subparser

    return subparsers


def main():

    parser = get_parser()
    subparsers = get_subparsers(parser)

    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    # Does the use want to install?
    command = args.command

    from penrose.logger import bot
    from penrose.version import __version__

    # The user may just want to parse the version
    if command == "version":
        print(__version__)
        sys.exit(0)

    bot.info("Penrose Python Version: %s" % __version__)

    if command == "install":
        from .install import main

    elif command == "list":
        from .list import main

    # No argument supplied, list packages
    else:
        from .list import main

    # Pass on to the correct parser
    if command is not None:
        main(args=args,
             parser=parser)


if __name__ == '__main__':
    main()
