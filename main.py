import argparse
import sys
import torchlight

from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    processors = dict()
    processors['stgcn_swmv'] = import_class('tools.recognition.REC_Processor')
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])
    arg = parser.parse_args()
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    p.start()