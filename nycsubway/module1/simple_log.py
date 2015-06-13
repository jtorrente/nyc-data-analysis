"""
    Copyright Javier Torrente (contact@jtorrente.info), 2015.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

    ************************************************************************

    NOTE: I developed the program contained in this file 'simple_log.py' as
    part of Udacity's nano-degree in data science. This file contains the
    code for the first project that I submitted after module 1. The code
    is publicly available in case it is useful, but if you happen to reuse
    it in your own nanodegree projects, do not forget to indicate that -
    otherwise it may be considered as cheating!

    ************************************************************************

    The source code in this file is entirely my own creation. However, source
    code contained in other parts of this project may contain contributions
    from Udacity's team. This is the case of module "problemsets", as in there
    I've put all the code corresponding to problem sets 1-4 in the "Introduction
    to Data Science" course. Udacity provides the code half completed - the
    student (me in this case) is then instructed to complete it. Therefore
    the code contained in problemsets cannot be considered my sole contribution.
"""
__author__ = 'jtorrente'

class SimpleLog:
    """
    Simple class used for logging messages into console.
    It is used to give a sense of progress, as it remembers
    total number of steps so it can inform the user how much
    it remains
    """
    total_steps = 0
    current_step = 0

    def __init__(self, _total_steps):
        self.total_steps = _total_steps

    def split_in_lines(self, message):
        """
        Splits the given message in lines, using the character \n as line separator.
        The message has to end with a \n char, otherwise last sentence will not be considered
        Example: message = "Line 1\nLine2\n\nLine3"
        This returns ["Line 1", "Line2", ""]
        :param message: The sentence to split in lines
        :return:    (1) The max number of characters in one line (max line length) as an integer and
                    (2) A list with all the lines
        """
        total_length = len(message)
        i = total_length
        line = ""
        max_chars = -1
        all_lines = []
        while i > 0:
            current_char = message[total_length-i]
            if current_char == '\n':
                if len(line)>max_chars:
                    max_chars = len(line)
                all_lines.append(line)
                line = ""
            else:
                line = line + current_char
            i -= 1
        return max_chars, all_lines

    def log_object(self, obj, title):
        """
        Logs on console the given title (String) and then prints the object
        It is equivalent to invoking log(title) and then printing the object
        :type title: string
        :param obj: The object to print
        :param title: The title that precedes the object
        """
        self.log(title + ("\n" if title[:len(title)] != '\n' else ""))
        print obj
        print ""

    def log(self, message):
        """
        Logs the given message on screen. The message can contain several lines separated by character '\n'
        The message will be formatted as follows:
        ----------------------
        |  STEP X/TOTAL...
        ----------------------
        | Line 1
        | Line 2
        | ...
        | Line N
        ----------------------

        :param message: The message to print. Can contain lines. Example: "Line 1\nLine 2\nLine3\n.
                        It must terminate in \n, otherwise last line is not printed
        :return: Nothing
        """
        self.current_step += 1
        total_message = r"   STEP " + str(self.current_step) + r"/" + str(self.total_steps) + r"..." + "   \n" + message
        first_line = ""
        max_chars_per_line, all_lines = self.split_in_lines(total_message)
        for i in range (0, max_chars_per_line + 4):
            first_line += "-"
        print first_line
        i=0
        for line in all_lines:
            print "| " + line
            if i == 0:
                print first_line
            i += 1
        if len(all_lines) != 1:
            print first_line
