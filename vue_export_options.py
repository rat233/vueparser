"""Shared model export switches for both command-line entry points."""
from dataclasses import dataclass
import argparse
import math

# Full-circle density, following rvmparser's fixed tessellation strategy.
# Keep the requested export range and 8-segment alignment for full circles.
SEGMENTS_MINIMUM = 16
SEGMENTS_MAXIMUM = 256
SEGMENTS_ALIGNMENT = 8
DEFAULT_SEGMENTS = 16
SEGMENTATION_POLICY_VERSION = 'fixed-8-aligned-v2'
_SEGMENT_EXAMPLES = '16, 24, 32, 40'


def validate_segments(value):
    """Return ``value`` as an int, or raise ValueError describing the policy.

    Shared by ``ExportOptions`` (direct Python exports) and the ``--segments``
    CLI converter so both entry points enforce the same constraints early.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            'segments must be an integer multiple of %d in [%d, %d] (e.g. %s); got %r'
            % (SEGMENTS_ALIGNMENT, SEGMENTS_MINIMUM, SEGMENTS_MAXIMUM,
               _SEGMENT_EXAMPLES, value))
    if not SEGMENTS_MINIMUM <= value <= SEGMENTS_MAXIMUM:
        raise ValueError(
            'segments must be in [%d, %d] (e.g. %s); got %d'
            % (SEGMENTS_MINIMUM, SEGMENTS_MAXIMUM, _SEGMENT_EXAMPLES, value))
    if value % SEGMENTS_ALIGNMENT:
        raise ValueError(
            'segments must be a multiple of %d (e.g. %s); got %d'
            % (SEGMENTS_ALIGNMENT, _SEGMENT_EXAMPLES, value))
    return int(value)


def parse_segments(text):
    """Argparse type for ``--segments``: parse then apply the shared policy."""
    try:
        value = int(text, 10)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            'segments must be an integer multiple of %d in [%d, %d] (e.g. %s); got %r'
            % (SEGMENTS_ALIGNMENT, SEGMENTS_MINIMUM, SEGMENTS_MAXIMUM,
               _SEGMENT_EXAMPLES, text))
    try:
        return validate_segments(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


# Use the same full-circle closure tolerance throughout geometry generation.
FULL_TURN_TOLERANCE = 1e-8


def is_full_turn(angle):
    """True when ``angle`` is a full turn within ``FULL_TURN_TOLERANCE``."""
    return abs(abs(float(angle)) - 2 * math.pi) <= FULL_TURN_TOLERANCE


def normalize_sweep(angle):
    """Return the signed exact full turn inside the closure tolerance.

    Sweeps outside the tolerance are returned unchanged; consumers must keep
    their native sweep and the proportional path rule for partial arcs.
    """
    value = float(angle)
    if is_full_turn(value):
        return math.copysign(2 * math.pi, value)
    return value


def full_turn_chord_error(radius, segments):
    """Maximum radial gap between a circle and its inscribed polygon."""
    return float(radius) * (1.0 - math.cos(math.pi / int(segments)))


def arc_path_segments(angle, segments):
    """rvmparser fixed mode: proportional arc divisions, at least three."""
    if is_full_turn(angle):
        return max(3, int(segments))
    return max(3, int(math.ceil(abs(angle) / (2 * math.pi) * segments)))


@dataclass(frozen=True)
class ExportOptions:
    format: str = 'all'
    reports: bool = True  # Existing Python research callers retain their reports.
    merge_moniker: bool = False
    fbx_mode: str = 'auto'
    mdb2: str | None = None  # Python research callers opt in explicitly.

    glb_up_axis: str = 'y'  # Standard GLB Y-up, consistent with the CLI default.
    unknown_bbox_wireframe: bool = True  # Preserve diagnostic placeholders by default.
    # 0928 review fix: workload/scheduling budget = summed geometry-split
    # scratch ESTIMATE limit (also caps per-batch split scratch). NOT a
    # process memory cap: resident data + in-flight batch + output encode and
    # subprocess memory live within and beyond this number.
    scratch_budget_mb: int | None = None  # None: unlimited.
    # P1 (0928 plan §2.2): circumferential segments for the exportable circular
    # primitives.  Single shared default for CLI and direct Python exports.
    segments: int = DEFAULT_SEGMENTS

    def __post_init__(self):
        if self.glb_up_axis not in ('y','z'):raise ValueError('invalid GLB up axis')
        if self.format not in ('all','glb','fbx','obj'):raise ValueError('invalid output format')
        if self.fbx_mode not in ('auto','glb'):raise ValueError('invalid FBX mode')
        if self.scratch_budget_mb is not None and self.scratch_budget_mb <= 0:
            raise ValueError('scratch budget must be a positive number of MiB')
        validate_segments(self.segments)

    @property
    def formats(self):
        return ('glb','obj','fbx') if self.format=='all' else (self.format,)


def parse_boolean(value):
    normalized = value.lower()
    if normalized == 'true':
        return True
    if normalized == 'false':
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def add_export_arguments(parser):
    metadata = parser.add_mutually_exclusive_group()
    metadata.add_argument('--mdb2', default='auto', help='MDB2 path, or auto to discover a same-stem database (default). GLB only.')
    metadata.add_argument('--no-mdb2', dest='mdb2', action='store_const', const=None, help='Export geometry without MDB2 semantics.')
    parser.add_argument('--format',choices=('all','glb','fbx','obj'),default='all',help='Model formats to publish (default: all).')
    reports = parser.add_mutually_exclusive_group()
    reports.add_argument('--reports',type=parse_boolean,metavar='{true,false}',default=False,
                         help='Write CSV/JSON diagnostic reports (default: false).')
    reports.add_argument('--no-reports',dest='reports',action='store_false',help=argparse.SUPPRESS)
    merge = parser.add_mutually_exclusive_group()
    merge.add_argument('--merge-moniker','--merge-components',type=parse_boolean,
                       metavar='{true,false}',default=False,
                       help='Merge all parts of each Moniker into one mesh with material slots (default: false).')
    merge.add_argument('--no-merge-moniker','--no-merge-components',dest='merge_moniker',
                       action='store_false',help=argparse.SUPPRESS)
    parser.add_argument('--glb-up-axis',choices=('y','z'),default='y',help='GLB vertical axis: y for standard viewers (default), z for source coordinates. FBX/OBJ unchanged.')
    parser.add_argument('--fbx-mode',choices=('glb','auto'),default='glb',help='FBX conversion route: glb (default), or ASCII with GLB fallback; diagnostic lines use GLB. Ignored without FBX.')
    unknown = parser.add_mutually_exclusive_group()
    unknown.add_argument('--unknown-bbox-wireframe',type=parse_boolean,metavar='{true,false}',
                         default=True,help='Include magenta line bounding boxes for unrecognized geometry (default: true).')
    unknown.add_argument('--no-unknown-bbox-wireframe',dest='unknown_bbox_wireframe',
                         action='store_false',help=argparse.SUPPRESS)
    parser.add_argument('--scratch-budget-mb',type=int,default=None,
                        help='Geometry split scratch workload budget in MiB: exports fail early when the summed split scratch ESTIMATE exceeds it, and split batches never exceed it (default: unlimited). Scheduling budget, not a hard process-memory cap.')
    parser.add_argument('--segments',type=parse_segments,default=DEFAULT_SEGMENTS,metavar='N',
                        help='Circumferential segments for circular primitives (cylinder, torus, cone, ellipsoid head, revolved profile, arc/weld tube). Must be an integer multiple of %d in [%d, %d]; default %d (e.g. 16, 24, 32, 40).' % (SEGMENTS_ALIGNMENT, SEGMENTS_MINIMUM, SEGMENTS_MAXIMUM, DEFAULT_SEGMENTS))


def options_from_args(args):
    return ExportOptions(args.format,args.reports,args.merge_moniker,args.fbx_mode,args.mdb2,
                         args.glb_up_axis,args.unknown_bbox_wireframe,args.scratch_budget_mb,
                         args.segments)
