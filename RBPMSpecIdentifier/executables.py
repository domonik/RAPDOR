import argparse
from RBPMSpecIdentifier.datastructures import _analysis_executable_wrapper, RBPMSpecData
from RBPMSpecIdentifier.visualize.dashboard import _gui_wrapper



def _add_common_args(parser):
    parser.add_argument(
        '--input',
        type=str,
        help="Path to the csv file containing protein counts as well as any additional information",
        required=True
    )
    parser.add_argument(
        '--sep',
        type=str,
        help="Seperator of the csv files (must be the same for the data and experimental design)",
        default="\t"
    )
    parser.add_argument(
        '--design-matrix',
        type=str,
        help="Design matrix specifying which columns in the --input contain the count data",
        required=True,
    )
    parser.add_argument(
        '--logbase',
        type=int,
        default=None,
        help="If input counts are log transformed please set the log base via this flag"
    )
    return parser



def analyze_parser(subparsers, name):
    parser = subparsers.add_parser(
        name,
        description="Runs the main RBPMSpecIdentifier Tool"
    )
    parser = _add_common_args(parser)

    parser.add_argument(
        '--distance-method',
        type=str,
        default="jensenshannon",
        help=f"Distance Method to use for calculation of sample differences. Can be one of {RBPMSpecData.methods}"
    )
    parser.add_argument(
        '--kernel-size',
        type=int,
        default=3,
        help=f"Uses an averaging kernel to run over fractions. This usually o stabilizes between sample variance."
             f" Set to 0 to disable this"
    )
    return parser


def gui_parser(subparsers, name):
    parser = subparsers.add_parser(
        name,
        description="Runs the RBPMSpecIdentifier GUI"
    )
    parser = _add_common_args(parser)
    parser.add_argument(
        '--port',
        type=str,
        help="Port to run the Dash server (Default: 8080)",
        default="8080"
    )
    parser.add_argument(
        '--host',
        type=str,
        help="Host IP used by the dash server to serve the application (Default: 0.0.0.0)",
        default="0.0.0.0"
    )
    parser.add_argument(
        '--debug',
        action="store_true",
        help="Runs dashboard in debug mode",
    )
    return parser



class RBPMSpecIdentifier:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            "RBPMSpecIdentifier suite",
            usage="RBPMSpecIdentifier <command> [<args>]"

        )
        self.methods = {
            #"visualize": (visualization_parser, run_visualization),
            "analyze": (analyze_parser, _analysis_executable_wrapper),
            "gui": (gui_parser, _gui_wrapper),
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
    RBPMSpecIdentifier().run()

def documentation_wrapper():
    parser = RBPMSpecIdentifier().parser
    return parser


if __name__ == '__main__':
    main()