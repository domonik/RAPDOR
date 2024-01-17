import argparse
import os
import codecs
from RAPDOR.datastructures import _analysis_executable_wrapper, RAPDORData
from RAPDOR.visualize.runApp import _gui_wrapper


def unescaped_str(arg_str):
    return codecs.decode(str(arg_str), 'unicode_escape')

def _add_common_args(parser, needed: bool = True):
    parser.add_argument(
        '--input',
        type=str,
        help="Path to the csv file containing protein counts as well as any additional information",
        required=needed,
        default=None
    )
    parser.add_argument(
        '--sep',
        type=unescaped_str,
        help="Seperator of the csv files (must be the same for the data and experimental design)",
        default="\t"
    )
    parser.add_argument(
        '--design-matrix',
        type=str,
        help="Design matrix specifying which columns in the --input contain the count data",
        required=needed,
        default=None
    )
    parser.add_argument(
        '--logbase',
        type=int,
        default=None,
        help="If input counts are log transformed please set the log base via this flag"
    )
    return parser

def _qtparser(subparsers, name):
    parser = subparsers.add_parser(
        name,
        description="Runs the GUI RAPDOR Tool"
    )
    return parser


def _analyze_parser(subparsers, name):
    parser = subparsers.add_parser(
        name,
        description="Runs the main RAPDOR Tool"
    )
    parser = _add_common_args(parser)

    parser.add_argument(
        '--distance-method',
        type=str,
        default="Jensen-Shannon-Distance",
        help=f"Distance Method to use for calculation of sample differences. Can be one of {RAPDORData.methods}"
    )
    parser.add_argument(
        '--kernel-size',
        type=int,
        default=3,
        help=f"Uses an averaging kernel to run over fractions. This usually o stabilizes between sample variance."
             f" Set to 0 to disable this"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method used for p-value calculation. One of ANOSIM or PERMANOVA. "
             "Default is None and will skip p-Value calculation"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0,
        help="Epsilon added to counts. This is useful if kl-divergence is used for distance calculations since it will"
             "result in nan for zero counts"
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Number of permutations used for ANOSIM or PERMANOVA."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output table file to store results"
    )
    parser.add_argument(
        "--global-permutation",
        action="store_true",
        help="Will use all proteins as background in ANOSIM or PERMANOVA p-value calculation. This might be unreliable"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Will write an output json file"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=os.cpu_count(),
        help="Number of cores used for permutation analysis. Per default uses all cores"
    )
    return parser


def _dash_parser(subparsers, name):
    parser = subparsers.add_parser(
        name,
        description="Runs the RAPDOR GUI"
    )
    parser = _add_common_args(parser, needed=False)
    parser.add_argument(
        '--port',
        type=str,
        help="Port to run the Dash server (Default: 8080)",
        default="8080"
    )
    parser.add_argument(
        '--host',
        type=str,
        help="Host IP used by the dash server to serve the application (Default:127.0.0.1)",
        default="127.0.0.1"
    )
    parser.add_argument(
        '--debug',
        action="store_true",
        help="Runs dashboard in debug mode",
    )
    return parser



class RAPDOR:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "RAPDOR suite",
            usage="RAPDOR <command> [<args>]"

        )
        self.methods = {
            "Analyze": (_analyze_parser, _analysis_executable_wrapper),
            "Dash": (_dash_parser, _gui_wrapper),
        }
        self.subparsers = self.parser.add_subparsers()
        self.__addparsers()

    def __addparsers(self):
        for name, (parser_add, func) in self.methods.items():
            subp = parser_add(self.subparsers, name)
            subp.set_defaults(func=func)

    def parse_args(self):
        args = self.parser.parse_args()
        return args

    def run(self):
        args = self.parse_args()
        args.func(args)


def main():
    RAPDOR().run()


def documentation_wrapper():
    parser = RAPDOR().parser
    return parser


if __name__ == '__main__':
    main()