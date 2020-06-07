"""[summary]

[description]
"""

import time
import argparse


__all__ = ['LoadInputFile', 'timeofday']


class LoadInputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """
        this overloads the template argparse.Action.__call__ method
        and must keep the same argument names
        """

        # argument values is an open file handle, with statement is just added
        # prettiness to make it clear that values is a file handle
        with values as fh:
            raw_lines = fh.read().splitlines()
            fh.close()

        stripped_lines = [rl.strip().split('#')[0] for rl in raw_lines]
        # the following strip blank lines, since '' evaluates to False
        input_lines = [sl for sl in stripped_lines if sl]
        args = []

        for line in input_lines:
            if line.startswith('<'):
                # These file lines are considered special just in case
                # we ever want to do something with these headers, like
                # spit the whole simulation config back out to stdout/log
                # like Athena does.
                #
                # line.strip()[1:-1]
                pass
            else:
                key, val = [kv.strip() for kv in line.strip().split('=')]
                args.append('--%s=%s' % (key, val))

        parser.parse_known_args(args, namespace)

        return


def timeofday():
    return time.strftime("%H:%M:%S")
