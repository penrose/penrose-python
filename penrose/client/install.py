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

#from penrose.main.validate import PenroseValidator
from penrose.logger import bot
import tempfile
import sys
import os


def main(args,parser,subparser):

    folder = args.folder
    if folder is None:
        folder = os.getcwd()

    source = args.src[0]
    if source is None:
        bot.error('Please provide a Github https address to install.')
        sys.exit(1)

    # Is the experiment valid?
    #TODO: Write validator here.
    #cli = PenroseValidator()
    #valid = cli.validate(source, cleanup=False)
    
    if valid is True:

        # Local Install
        print('todo: install (what does that mean?)')
