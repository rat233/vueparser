"""Audit root display matrices without claiming a complete nested transform parser.

G+0x78 and G+0xf8 are inverse 4x4 double matrices in all current root groups.
First direct 0x1f/0xd0 children provide a bounded geometry crosscheck. Exact
circular bounds and four-cardinal-point envelopes are distinguished; G+0x178 bounds must not be blindly transformed a second time.
"""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
import numpy as np
import vue_record_parser as vrp

ROOT=Path(__file__).resolve().parent


def read_matrix_pair(data, group):
    g=group['header_start']
    if g < 0 or g+0x178>group['end'] or group['end']>len(data):
        raise ValueError('Incomplete root matrix pair')
    a=np.array(struct.unpack_from('<16d',data,g+0x78)).reshape(4,4)
    b=np.array(struct.unpack_from('<16d',data,g+0xf8)).reshape(4,4)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Nonfinite root matrix pair')
    affine_error=float(max(np.max(np.abs(a[:,3]-[0,0,0,1])),np.max(np.abs(b[:,3]-[0,0,0,1]))))
    inverse_error=float(max(np.max(np.abs(a@b-np.eye(4))),np.max(np.abs(b@a-np.eye(4)))))
    if affine_error>1e-10 or inverse_error>1e-6:
        raise ValueError('Root matrix affine/inverse constraint failed')
    return a,b,inverse_error


def first_cylinder_check(data, group, matrix):
    p=group['header_start']+0x1e0
    if p+0xc8>group['end'] or data[p:p+8]!=struct.pack('<2I',0x1f,0xd0):
        return {'status':'not_supported_first_direct_cylinder'}
    axis=np.array(struct.unpack_from('<3d',data,p+0x50))
    point=np.array(struct.unpack_from('<3d',data,p+0x78))
    ref=np.array(struct.unpack_from('<3d',data,p+0xa8))
    bbox=np.array(struct.unpack_from('<6d',data,p+8))
    first=(np.r_[point,1]@matrix)[:3]
    second=(np.r_[point+axis,1]@matrix)[:3]
    direction=second-first
    length=np.linalg.norm(direction)
    if not np.isfinite(length) or length<=0:
        return {'status':'degenerate_candidate','tag_offset':p}
    radius=np.linalg.norm(ref@matrix[:3,:3])
    radial=radius*np.sqrt(np.maximum(0,1-(direction/length)**2))
    calculated=np.r_[np.minimum(first,second)-radial,np.maximum(first,second)+radial]
    error=float(np.max(np.abs(calculated-bbox)))
    unit=np.array(struct.unpack_from('<3d',data,p+0x90))
    first_radial=ref@matrix[:3,:3]
    second_radial=np.cross(unit,ref)@matrix[:3,:3]
    cardinal_extent=np.maximum(np.abs(first_radial),np.abs(second_radial))
    cardinal_box=np.r_[np.minimum(first,second)-cardinal_extent,np.maximum(first,second)+cardinal_extent]
    cardinal_error=float(np.max(np.abs(cardinal_box-bbox)))
    cached_bbox=np.array(struct.unpack_from('<6d',data,group['header_start']+0x178))
    rigid_error=float(np.max(np.abs(matrix[:3,:3]@matrix[:3,:3].T-np.eye(3))))
    status=('exact_cylinder_bbox_matches' if error<1e-6 and rigid_error<1e-6 else
            'cardinal_envelope_matches_only' if cardinal_error<1e-6 and rigid_error<1e-6 else
            'bbox_mismatch_or_nonrigid')
    return {'status':status,
            'tag_offset':p,'bbox_max_error_m':error,'rigid_error':rigid_error,
            'cardinal_envelope_error_m':cardinal_error,
            'group_178_bbox_candidate_vs_first_cylinder_error_m':float(np.max(np.abs(cached_bbox-bbox))),
            'bbox_warning':('Stored box uses only four cardinal directions per cap and can underbound a full circle'
                            if status=='cardinal_envelope_matches_only' else None),
            'point_world_candidate':first.tolist(),'end_world_candidate':second.tolist(),
            'scope':'Only this first direct cylinder; does not assign transforms to all descendants'}


def audit(path):
    data=path.read_bytes()
    _,records,errors=vrp.read_records(data)
    if errors:
        raise ValueError(str(errors))
    counts=Counter();details=[];max_inverse=0
    for record in records:
        for group in record['display_groups']:
            a,b,error=read_matrix_pair(data,group)
            identity=np.max(np.abs(a-np.eye(4)))<1e-10
            check=first_cylinder_check(data,group,a)
            counts['identity' if identity else 'nonidentity']+=1
            counts[('identity_' if identity else 'nonidentity_')+check['status']]+=1
            max_inverse=max(max_inverse,error)
            details.append({'record_offset':record['record_offset'],'moniker':record['moniker'],
                            'group_index':group['index'],'display_code':group['display_code_raw'],
                            'group_start':group['header_start'],'identity':bool(identity),
                            'forward_matrix_candidate_row_major':a.tolist(),'inverse_matrix_row_major':b.tolist(),
                            'inverse_max_error':error,'first_child_geometry_check':check})
    return {'sample':path.stem,'file_sha256':hashlib.sha256(data).hexdigest(),
            'summary':{'root_groups':len(details),'max_inverse_error':max_inverse,**counts},
            'convention':'For checked first cylinders, [x,y,z,1] @ forward; translation in last row',
            'limitations':['Nested transform binding/instancing not decoded',
                           'Matrix pair identity alone does not prove all geometry is in world coordinates',
                           'Do not transform a world bbox or already-world coordinates twice',
                           'Four-cardinal-point source boxes may underbound circles; recompute bounds from recovered geometry' ],
            'groups':details}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('vue',nargs='+',type=Path);args=ap.parse_args()
    out=ROOT/'outputs/vue_record_header';out.mkdir(parents=True,exist_ok=True)
    for path in args.vue:
        result=audit(path)
        (out/(path.stem+'_root_transforms.json')).write_text(json.dumps(result,ensure_ascii=False,indent=1),encoding='utf-8')
        print(path.stem,result['summary'])
