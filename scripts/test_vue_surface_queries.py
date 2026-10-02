"""Direct checks for P4 reference coverage and native arc placement bounds."""
from pathlib import Path
import struct
import unittest
import numpy as np
import vue_preview_export as v


class SurfaceQueryTests(unittest.TestCase):
    def test_batched_reference_points_and_local_origin(self):
        block=dict(point=np.zeros(3),axis=np.array([0.,0.,1.]),dirv=np.array([0.,0.,1.]),radius=1.)
        vertices,faces=v.cylinder(block,16)
        points=np.array([[1,0,.5],[0,1,.5],[0,0,0],[0,0,.5]])
        shift=np.array([2400.,1500.,12.])
        query=v.SurfaceQuery(vertices,faces,origin=shift)
        np.testing.assert_array_equal(query.contains_points(points+shift),[True,True,True,False])
        self.assertTrue(v.references_on_surfaces(points[:3]+shift,iter([query])))
        self.assertFalse(v.references_on_surfaces(points+shift,iter([query])))
        self.assertEqual(query.triangle_low.shape,(len(faces),3))

    def test_native_arc_bounds_match_dense_support(self):
        data=Path('15300.vue').read_bytes();off=3538660+1088
        center=v.float_view(data,off+0x50,3);normal=v.float_view(data,off+0x68,3)
        start=v.float_view(data,off+0x98,3);axis=v.float_view(data,off+0xb0,3)
        angle=struct.unpack_from('<d',data,off+0x80)[0];radius=np.linalg.norm(start-center)
        tube=v.arc_tube_radius_from_vue(data,off);u=(start-center)/radius
        matrix=np.eye(4)
        for sweep in (angle,-angle,2*np.pi):
            with self.subTest(sweep=sweep):
                t=np.linspace(0,sweep,32769)
                c=np.cos(t)[:,None];s=np.sin(t)[:,None]
                points=center+radius*(c*u+s*axis)
                tangent=-s*u+c*axis
                w1=np.broadcast_to(normal,points.shape);w2=np.cross(tangent,w1)
                w2/=np.linalg.norm(w2,axis=1)[:,None]
                dense=v.swept_tube_bounds(points,w1,w2,tube,matrix)
                actual=v.arc_tube_precise_bounds(center,start,axis,normal,tube,sweep,matrix)
                np.testing.assert_allclose(actual,dense,atol=2e-8,rtol=0)


if __name__=='__main__':unittest.main()
