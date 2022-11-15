import argparse
from utils.pretext.jigsaw import JigsawPretext
from utils.pretext.rotation import RotationPretext

def get_pretext_args(parser):
    tmp_parser = argparse.ArgumentParser()    
    tmp_parser.add_argument('--pretext', type=str, default='rotation', choices=['jigsaw', 'rotation'],
                        help='jigsaw|rotation|mixed')

    parser.add_argument('--pretext', type=str, default='rotation', choices=['jigsaw', 'rotation'],
                        help='jigsaw|rotation')
    args, _ = tmp_parser.parse_known_args()

    if args.pretext is None:
        return parser
    elif args.pretext == 'rotation':
        return RotationPretext.update_parser(parser)
    elif args.pretext == 'jigsaw':
        return JigsawPretext.update_parser(parser)
    else:
        raise ValueError('Unknown pretext task: {}'.format(args.pretext))


def get_pretext_task(args):
    if not hasattr(args, 'pretext'):
        return None
    elif args.pretext == 'rotation':
        return RotationPretext(args)
    elif args.pretext == 'jigsaw':
        return JigsawPretext(args)
    else:
        raise ValueError('Unknown pretext task: {}'.format(args.pretext))