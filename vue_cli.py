"""Standalone command-line entry for the current VUE geometry exporter."""
import argparse
from pathlib import Path
import sys
from vue_export_options import add_export_arguments, options_from_args

VERSION = '30.1'


def main():
    if len(sys.argv)>2 and sys.argv[1]=='--internal-helper':
        module=sys.argv[2]
        sys.argv=[sys.argv[0],*sys.argv[3:]]
        if module=='vue_fbx_phong_material':
            from vue_fbx_phong_material import main as helper
        elif module=='vue_fbx_from_exported_glb':
            from vue_fbx_from_exported_glb import main as helper
        else:
            raise SystemExit('Unknown internal helper')
        helper()
        return 0
    parser=argparse.ArgumentParser(prog='VueParser',description='Convert Smart 3D VUE geometry to GLB, OBJ and binary FBX (metres; GLB Y-up by default, FBX/OBJ Z-up).')
    parser.add_argument('inputs',nargs='+',type=Path,help='One or more .vue files; quote paths containing spaces.')
    parser.add_argument('-o','--output-dir',type=Path,default=Path('outputs'),help='Output root; each input gets a subfolder (default: outputs).')
    add_export_arguments(parser)
    parser.add_argument('--version',action='version',version='VueParser '+VERSION+' (geometry v29)')
    if len(sys.argv)==1:
        parser.print_help()
        return 0
    args=parser.parse_args()
    for source in args.inputs:
        if source.suffix.lower()!='.vue' or not source.is_file():
            parser.error('Input is not an existing .vue file: '+str(source))
    if args.mdb2 not in (None, 'auto') and len(args.inputs) != 1:
        parser.error('--mdb2 with an explicit path requires exactly one VUE input.')
    names=[p.stem.casefold() for p in args.inputs]
    if len(set(names))!=len(names):parser.error('Input names must be distinct to avoid overwriting output folders.')
    from vue_progress import console_progress
    console_progress('正在加载几何处理模块…')
    import vue_preview_export as exporter
    try:
        settings=options_from_args(args)
        for source in args.inputs:
            exporter.export_sample(source,args.output_dir/source.stem,settings,progress=console_progress)
    except KeyboardInterrupt:
        print('Cancelled.',file=sys.stderr);return 130
    except Exception as exc:
        print('Conversion failed: '+str(exc),file=sys.stderr);return 1
    return 0

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
