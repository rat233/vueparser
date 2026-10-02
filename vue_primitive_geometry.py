"""Preallocated topology and native normals for elementary VUE surfaces."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=128)
def ring_faces(rows, columns, closed=False):
    """Two triangles per cell, with a closed circumferential seam."""
    spans = rows if closed else rows - 1
    i = np.arange(spans)[:, None] * columns
    j = np.arange(columns)[None, :]
    a = i + j
    b = i + (j + 1) % columns
    d = ((i // columns + 1) % rows) * columns + j
    c = ((i // columns + 1) % rows) * columns + (j + 1) % columns
    faces = np.empty((spans, columns, 2, 3), dtype=np.int64)
    faces[:, :, 0, 0], faces[:, :, 0, 1], faces[:, :, 0, 2] = a, b, c
    faces[:, :, 1, 0], faces[:, :, 1, 1], faces[:, :, 1, 2] = a, c, d
    faces = faces.reshape(-1, 3)
    faces.setflags(write=False)
    return faces


def finish(vertices, faces, normals, matrix=None):
    """Transform row-vector positions/normals and align winding with the surface."""
    vertices = np.asarray(vertices, dtype=float)
    normals = np.asarray(normals, dtype=float)
    faces = np.array(faces, dtype=np.int64, copy=True)
    if matrix is not None:
        matrix = np.asarray(matrix).reshape(4, 4)
        vertices = vertices @ matrix[:3, :3] + matrix[3, :3]
        try:
            normals = normals @ np.linalg.inv(matrix[:3, :3]).T
        except np.linalg.LinAlgError:
            return vertices, faces, None
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    reverse = np.einsum('ij,ij->i', cross, normals[faces].sum(axis=1)) < 0
    faces[reverse] = faces[reverse, ::-1]
    return vertices, faces, normals


def planar_normals(vertices, faces):
    """One exact plane normal, using the largest face to avoid tiny triangles."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal = cross[np.argmax(np.einsum('ij,ij->i', cross, cross))]
    length = np.linalg.norm(normal)
    if length <= 1e-15:
        return None
    return np.broadcast_to(normal / length, vertices.shape)


def prism(bottom, top, cap_faces):
    """Polygonal prism with separate planar caps and four vertices per side."""
    bottom, top = np.asarray(bottom), np.asarray(top)
    n = len(bottom)
    cap_faces = np.asarray(cap_faces)
    vertices = np.empty((6 * n, 3), dtype=float)
    normals = np.empty_like(vertices)
    vertices[:n], vertices[n:2*n] = bottom, top
    cap_normal = planar_normals(bottom, cap_faces)[0]
    if cap_normal @ (top - bottom).mean(0) < 0:
        cap_normal = -cap_normal
    normals[:n], normals[n:2*n] = -cap_normal, cap_normal
    sides = vertices[2*n:].reshape(n, 4, 3)
    sides[:,0], sides[:,1] = bottom, np.roll(bottom, -1, axis=0)
    sides[:,2], sides[:,3] = np.roll(top, -1, axis=0), top
    side_normals = np.cross(sides[:,1]-sides[:,0], sides[:,3]-sides[:,0])
    # The cap polygon's signed area determines the direction for every side,
    # including concave profiles; a centroid test would fail in a concavity.
    area = np.cross(bottom - bottom[0], np.roll(bottom,-1,axis=0)-bottom[0]).sum(0)
    if area @ cap_normal < 0:
        side_normals = -side_normals
    normals[2*n:] = np.repeat(side_normals, 4, axis=0)
    faces = np.empty((2*len(cap_faces)+2*n, 3), dtype=np.int64)
    faces[:len(cap_faces)] = cap_faces
    faces[len(cap_faces):2*len(cap_faces)] = cap_faces + n
    base = 2*n + 4*np.arange(n)
    faces[2*len(cap_faces):] = (base[:,None,None] + np.array([[0,1,2],[0,2,3]])).reshape(-1,3)
    return finish(vertices, faces, normals)


def tube(centers, w1, w2, tangents, radius, columns, closed, matrix):
    """Analytic round sweep with hard caps, one allocation per output array."""
    centers, w1, w2, tangents = map(np.asarray, (centers, w1, w2, tangents))
    rows = len(centers)
    side_count = rows * columns
    count = side_count if closed else side_count + 2*columns + 2
    vertices = np.empty((count,3), dtype=float)
    normals = np.empty_like(vertices)
    q = np.arange(columns) * (2*np.pi/columns)
    radial = np.cos(q)[None,:,None]*w1[:,None,:] + np.sin(q)[None,:,None]*w2[:,None,:]
    normals[:side_count] = radial.reshape(-1,3)
    vertices[:side_count] = (centers[:,None,:]+radius*radial).reshape(-1,3)
    side_faces = ring_faces(rows,columns,closed)
    faces = np.empty((len(side_faces)+(0 if closed else 2*columns),3),dtype=np.int64)
    faces[:len(side_faces)] = side_faces
    if not closed:
        vertices[side_count:side_count+columns] = vertices[:columns]
        vertices[side_count+columns:-2] = vertices[side_count-columns:side_count]
        vertices[-2:] = centers[[0,-1]]
        normals[side_count:side_count+columns] = normals[-2] = -tangents[0]
        normals[side_count+columns:-2] = normals[-1] = tangents[-1]
        i = np.arange(columns);j=(i+1)%columns
        caps=faces[len(side_faces):].reshape(columns,2,3)
        caps[:,0,0],caps[:,0,1],caps[:,0,2]=count-2,side_count+j,side_count+i
        caps[:,1,0],caps[:,1,1],caps[:,1,2]=count-1,side_count+columns+i,side_count+columns+j
    return finish(vertices,faces,normals,matrix)


def linear_revolution(profile, origin, columns, hard_angle=30):
    """Rotate a (radius, height) polyline, smoothing only compatible joins."""
    profile=np.asarray(profile)
    delta=np.diff(profile,axis=0)
    # Profile normals have radial/axial coefficients (dz, -dr).
    slope=np.column_stack((delta[:,1],-delta[:,0]))
    slope/=np.linalg.norm(slope,axis=1)[:,None]
    first,last=slope.copy(),slope.copy()
    smooth=np.einsum('ij,ij->i',slope[:-1],slope[1:])>=np.cos(np.deg2rad(hard_angle))
    average=slope[:-1]+slope[1:]
    average/=np.maximum(np.linalg.norm(average,axis=1)[:,None],1e-30)
    last[:-1][smooth]=average[smooth]
    first[1:][smooth]=average[smooth]
    angle=np.arange(columns)*2*np.pi/columns
    radial=np.column_stack((np.cos(angle),np.sin(angle),np.zeros(columns)))
    # Profile-edge loops are small; all circumferential work is vectorized.
    # Reuse a ring at smooth joins, duplicate only at hard profile corners.
    rings=[];normal_coeff=[];starts=[];counts=[];total=0
    for edge in range(len(delta)):
        pair=[]
        for end,coeff in ((edge,first[edge]),(edge+1,last[edge])):
            if end==edge and edge and smooth[edge-1]:
                pair.append(rings[-1][1]);continue
            count=1 if abs(profile[end,0])<=1e-10 and abs(coeff[0])<=1e-10 else columns
            pair.append(len(starts));starts.append(total);counts.append(count)
            normal_coeff.append((end,coeff));total+=count
        rings.append(pair)
    vertices=np.empty((total,3));normals=np.empty_like(vertices)
    for start,count,(end,coeff) in zip(starts,counts,normal_coeff):
        vertices[start:start+count]=origin+profile[end,0]*radial[:count]+[0,0,profile[end,1]]
        normals[start:start+count]=coeff[0]*radial[:count]+[0,0,coeff[1]]
    pole=np.abs(profile[:,0])<=1e-10
    face_count=sum(columns*(1 if pole[i] or pole[i+1] else 2) for i in range(len(delta)))
    faces=np.empty((face_count,3),dtype=np.int64);offset=0
    i=np.arange(columns);j=(i+1)%columns
    for edge,(left,right) in enumerate(rings):
        a=starts[left]+(np.zeros(columns,dtype=int) if counts[left]==1 else i)
        b=starts[left]+(np.zeros(columns,dtype=int) if counts[left]==1 else j)
        c=starts[right]+(np.zeros(columns,dtype=int) if counts[right]==1 else j)
        d=starts[right]+(np.zeros(columns,dtype=int) if counts[right]==1 else i)
        if pole[edge]:
            faces[offset:offset+columns]=np.column_stack((a,c,d));offset+=columns
        elif pole[edge+1]:
            faces[offset:offset+columns]=np.column_stack((a,b,d));offset+=columns
        else:
            block=faces[offset:offset+2*columns].reshape(columns,2,3)
            block[:,0]=np.column_stack((a,b,c));block[:,1]=np.column_stack((a,c,d));offset+=2*columns
    return finish(vertices,faces,normals)
