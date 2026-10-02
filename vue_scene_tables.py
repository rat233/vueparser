"""Bounded native VUE discipline/type, appearance and spatial table reader.

Scope: directory layout observed in 063E.vue and 15300.vue. Structural failures
raise ValueError; raw entries remain available. These indices are not MDB hierarchy.
"""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
import vue_record_parser as vrp

WORKSPACE = Path(__file__).resolve().parent
DIRECTORY = {'type_buckets': 0x52c, 'type_entries': 0x530,
             'type_strings': 0x534, 'type_strings_end': 0x538,
             'spatial_start': 0x53c, 'spatial_end': 0x540,
             'material_start': 0x548, 'material_end': 0x54c,
             'names': 0x558, 'discipline_roots': 0x564,
             'first_record': 0x588, 'type_count_plus_one': 0x594,
             'name_count': 0x5a4, 'record_count_plus_one': 0x5a8}


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Reader:
    def __init__(self, data):
        self.data = data

    def unpack(self, fmt, offset):
        require(0 <= offset <= len(self.data)-struct.calcsize(fmt),
                f'Out of bounds {fmt} at {offset:#x}')
        return struct.unpack_from(fmt, self.data, offset)

    def u32(self, offset):
        return self.unpack('<I', offset)[0]

    def string(self, offset, lo, hi):
        require(0 <= lo <= offset < hi <= len(self.data), 'String pointer outside pool')
        end = self.data.find(b'\0', offset, hi)
        require(end >= 0, 'Unterminated string in pool')
        return self.data[offset:end].decode('utf-8', 'strict')


def table_offsets(lo, hi, stride, file_size):
    require(0 <= lo < hi <= file_size and (hi-lo) % stride == 0,
            f'Invalid table extent [{lo:#x},{hi:#x}) stride {stride}')
    return range(lo, hi, stride)


def walk_records(head, by_start, field):
    visited, chain = set(), []
    while head:
        require(head in by_start, f'Unknown record start {head:#x} in {field}')
        require(head not in visited, f'Cyclic record chain in {field}')
        visited.add(head)
        record = by_start[head]
        chain.append(head)
        head = record[field]
    return chain


def contains(outer, inner, tolerance=1e-8):
    return all(outer[i]-tolerance <= inner[i] and
               inner[i+3] <= outer[i+3]+tolerance for i in range(3))


def read_scene_tables(data, records=None):
    reader = Reader(data)
    u = reader.u32
    if records is None:
        _, records, errors = vrp.read_records(data)
        require(not errors, str(errors))
    by_start = {r['record_offset']: r for r in records}
    require(len(by_start) == len(records) > 0, 'Missing/duplicate records')
    directory = {name: u(offset) for name, offset in DIRECTORY.items()}
    require(directory['first_record'] == min(by_start), 'Directory first record mismatch')
    require(directory['record_count_plus_one'] == len(records)+1, 'Directory count mismatch')
    n_names = directory['name_count']
    require(1 < n_names <= 1000, 'Unsupported discipline count')
    names_start, names_end = directory['names'], directory['discipline_roots']
    names = [reader.string(u(names_start+4*i), names_start+4*n_names, names_end)
             for i in range(n_names)]
    types, type_entries_seen, disciplines = {}, set(), []
    for index in range(1, n_names):
        bucket = u(directory['type_buckets']+4*index)
        require(directory['type_buckets']+4*n_names <= bucket and
                bucket+8 <= directory['type_entries'], 'Type bucket outside table')
        entry, count = reader.unpack('<2I', bucket)
        local = []
        for _ in range(count):
            require(directory['type_entries'] <= entry and entry+16 <= directory['type_strings']
                    and (entry-directory['type_entries']) % 16 == 0, 'Type entry outside grid')
            require(entry not in type_entries_seen, 'Repeated/cyclic type entry')
            type_entries_seen.add(entry)
            nxt, name_ptr, type_id, ordinal = reader.unpack('<4I', entry)
            label = reader.string(name_ptr, directory['type_strings'], directory['type_strings_end'])
            key = (index, type_id)
            require(key not in types, 'Duplicate discipline/type ID')
            item = {'entry_offset': entry, 'next_ptr': nxt, 'name_ptr': name_ptr,
                    'type_id': type_id, 'ordinal': ordinal, 'name': label,
                    'raw_hex': data[entry:entry+16].hex()}
            types[key] = item
            local.append(item)
            entry = nxt
        require(entry == 0, 'Type count / null tail mismatch')
        chain = walk_records(u(directory['discipline_roots']+4*index), by_start, 'field_50_raw')
        local_ids = [u(p+0x54) for p in chain]
        require(sorted(local_ids) == list(range(len(chain))), 'Discipline ordinal is not a permutation')
        disciplines.append({'index': index, 'name': names[index], 'types': local,
                            'record_offsets': chain, 'count': len(chain),
                            'field_54_is_local_ordinal_permutation': True})
    require(len(type_entries_seen)+1 == directory['type_count_plus_one'], 'Type total mismatch')
    require(type_entries_seen == set(table_offsets(directory['type_entries'],
            directory['type_strings'], 16, len(data))), 'Unvisited type entries')
    discipline_by_record = {}
    for discipline in disciplines:
        for offset in discipline['record_offsets']:
            require(offset not in discipline_by_record, 'Record in multiple discipline lists')
            discipline_by_record[offset] = discipline['index']
    require(set(discipline_by_record) == set(by_start), 'Discipline list coverage mismatch')

    materials = {}
    for index, offset in enumerate(table_offsets(directory['material_start'],
                directory['material_end'], 200, len(data))):
        raw = data[offset:offset+200]
        occupied = any(raw)
        item = {'offset': offset, 'slot_index': index, 'occupied': occupied, 'raw_hex': raw.hex()}
        if occupied:
            require(u(offset+0x1c) == index, 'Material slot self index mismatch')
            code = u(offset)
            item.update(packed_code_raw=code, discipline_index=u(offset+4),
                        type_id_candidate=(code & 0x1ff)//2,
                        color_index_candidate=code >> 9,
                        packed_color_raw=u(offset+0x24), packed_flags_raw=u(offset+0x28),
                        field_c4_raw=u(offset+0xc4),
                        diffuse_rgb=reader.unpack('<3f', offset+0x30),
                        specular_rgb=reader.unpack('<3f', offset+0x3c),
                        ambient_rgb=reader.unpack('<3f', offset+0x48),
                        opacity_candidate=reader.unpack('<f', offset+0x54)[0])
            require(all(math.isfinite(x) and 0 <= x <= 1 for x in (*item['diffuse_rgb'], item['opacity_candidate'])),
                    'Invalid native material RGBA')
            # Observed native alpha: 1 for opaque materials, 0.5 for translucent walls.
            # Keep the historical candidate field for existing reports/readers.
            item['opacity'] = item['opacity_candidate']
            item['transparency'] = 1.0 - item['opacity']
            require(1 <= item['discipline_index'] < n_names, 'Invalid material discipline index')
            require(reader.unpack('<3f',offset+0x58) == item['diffuse_rgb'] and
                    reader.unpack('<f',offset+0x64)[0] == item['opacity_candidate'],
                    'Duplicated material values mismatch')
        materials[offset] = item

    nodes = {}
    for offset in table_offsets(directory['spatial_start'], directory['spatial_end'], 120, len(data)):
        require(u(offset) == 8, 'Unsupported spatial node tag')
        bbox, center = reader.unpack('<6d', offset+8), reader.unpack('<3d', offset+0x60)
        require(all(math.isfinite(x) for x in (*bbox, *center)), 'Nonfinite spatial bounds')
        require(all(abs(center[i]-(bbox[i]+bbox[i+3])/2) < 1e-8 for i in range(3)), 'Spatial center mismatch')
        sides = [bbox[i+3]-bbox[i] for i in range(3)]
        require(min(sides) > 0 and max(sides)-min(sides) < 1e-8, 'Non-cubic spatial node')
        nodes[offset] = {'offset': offset, 'bbox_world': bbox, 'center': center,
                         'children': reader.unpack('<8I',offset+0x3c),
                         'object_lists': [walk_records(u(offset+f),by_start,'next_ptr_raw') for f in (4,0x38)],
                         'raw_hex': data[offset:offset+120].hex()}
    parents, spatial_owners, child_octants = {}, {}, {}
    for offset, node in nodes.items():
        for slot, child in enumerate(node['children']):
            if not child:
                continue
            require(child in nodes and child not in parents, 'Unknown/shared spatial child')
            parents[child] = offset
            box = nodes[child]['bbox_world']
            require(contains(node['bbox_world'],box), 'Child outside parent cell')
            require(all(abs((box[i+3]-box[i])*2-(node['bbox_world'][i+3]-node['bbox_world'][i])) < 1e-8
                        for i in range(3)), 'Child is not half-sized')
            octant = tuple(int(box[i] >= node['center'][i]-1e-8) for i in range(3))
            require(slot not in child_octants or child_octants[slot] == octant, 'Octant slot inconsistency')
            child_octants[slot] = octant
        for list_index, chain in enumerate(node['object_lists']):
            for p in chain:
                require(p not in spatial_owners, 'Record in multiple spatial lists')
                require(u(p+0x4c) == offset, 'Spatial owner back reference mismatch')
                require(contains(node['bbox_world'],by_start[p]['bbox_world']), 'Object outside owning cell')
                spatial_owners[p] = (offset,list_index)
    roots = set(nodes)-set(parents)
    require(roots == {directory['spatial_start']}, 'Unexpected spatial roots')
    visited, pending = set(), list(roots)
    while pending:
        p = pending.pop()
        require(p not in visited, 'Spatial cycle')
        visited.add(p)
        pending.extend(c for c in nodes[p]['children'] if c)
    require(visited == set(nodes), 'Unreachable spatial nodes')
    require(set(spatial_owners) == set(by_start), 'Spatial record coverage mismatch')

    decoded = []
    for p, record in sorted(by_start.items()):
        material_ptr = u(p+0x3c)
        require(material_ptr in materials and materials[material_ptr]['occupied'], 'Unresolved material pointer')
        mat = materials[material_ptr]
        discipline = discipline_by_record[p]
        type_id = data[p+0x77]+1
        require(mat['discipline_index'] == discipline, 'Material / chain discipline mismatch')
        require((discipline,type_id) in types, 'Unresolved native type ID')
        require(mat['packed_code_raw'] & 0x1ff == type_id*2, 'Material / header type mismatch')
        require(u(p+0x90) == u(p+0xac), 'Global ordinal duplicate mismatch')
        decoded.append({'record_offset':p,'moniker':record['moniker'],
                        'discipline_index':discipline,'discipline_name':names[discipline],
                        'type_id':type_id,'type_name':types[discipline,type_id]['name'],
                        'type_entry_offset':types[discipline,type_id]['entry_offset'],
                        'material_offset':material_ptr,'material_slot':mat['slot_index'],
                        'diffuse_rgb':mat['diffuse_rgb'],'specular_rgb':mat['specular_rgb'],
                        'ambient_rgb':mat['ambient_rgb'],'opacity_candidate':mat['opacity_candidate'],
                        'opacity':mat['opacity'],'transparency':mat['transparency'],
                        'spatial_node_offset':spatial_owners[p][0], 'spatial_list_index':spatial_owners[p][1],
                        'discipline_local_ordinal':u(p+0x54),'global_ordinal':u(p+0x90),
                        'next_in_discipline':u(p+0x50),'next_in_spatial_list':u(p+4)})
    require(sorted(r['global_ordinal'] for r in decoded) == list(range(len(records))), 'Global ordinal permutation mismatch')
    return {'file_sha256':hashlib.sha256(data).hexdigest(), 'file_bytes':len(data),
            'scope':'Observed directory layout in two samples; native indices are not MDB hierarchy',
            'directory':directory, 'model_name':names[0], 'disciplines':disciplines,
            'reference_rules':[
                {'source':'record+0x3c','interpretation':'absolute','target':'material entry',
                 'target_base':directory['material_start'],'stride':200,'checks':'bounds/grid/occupied/self slot/type/discipline'},
                {'source':'record+0x4c','interpretation':'absolute','target':'spatial node',
                 'target_base':directory['spatial_start'],'stride':120,'checks':'bounds/grid/tag/backlink/bbox/octree reachability'},
                {'source':'record+0x50','interpretation':'absolute','target':'next record start in discipline',
                 'checks':'native root / no cycles / exact coverage / material discipline'},
                {'source':'material+0x04','interpretation':'table_index','target':'native discipline name and root',
                 'target_base':directory['names'],'stride':4},
                {'source':'record+0x77 + 1','interpretation':'key_within_discipline_type_bucket',
                 'target':'native type entry; not global type ID or byte address'},
                {'source':'record+0x54','interpretation':'discipline_local_ordinal',
                 'checks':'exact 0..discipline count-1 permutation'},
                {'source':'record+0x90 and +0xac','interpretation':'global_ordinal',
                 'checks':'equal and exact 0..record count-1 permutation'},
                {'source':'root-list first,last,next','interpretation':'partition_relative',
                 'target_base':'record+0x70','checks':'display-group tag/count/tail/range via shared parser'}],
            'materials':list(materials.values()),'spatial_nodes':list(nodes.values()),
            'records':decoded,
            'summary':{'records':len(records),'disciplines':{x['name']:x['count'] for x in disciplines},
                       'types':len(type_entries_seen),'material_slots':len(materials),
                       'occupied_material_slots':sum(x['occupied'] for x in materials.values()),
                       'referenced_material_slots':len({x['material_offset'] for x in decoded}),
                       'spatial_nodes':len(nodes),'spatial_object_lists':sum(bool(c) for n in nodes.values() for c in n['object_lists']),
                       'spatial_child_octants':child_octants,
                       'record_opacity_candidates':dict(Counter(x['opacity_candidate'] for x in decoded)),
                       'all_structural_checks_passed':True}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('vue',nargs='+',type=Path)
    ap.add_argument('--output-dir',type=Path,default=WORKSPACE/'outputs'/'vue_record_header')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summaries = []
    for path in args.vue:
        report = read_scene_tables(path.read_bytes())
        report['source_file'] = str(path.resolve())
        (args.output_dir/(path.stem+'_scene_tables.json')).write_text(json.dumps(report,ensure_ascii=False,indent=1),encoding='utf-8')
        summary = {'sample':path.stem,'file_sha256':report['file_sha256'],**report['summary']}
        summaries.append(summary)
        print(json.dumps(summary,ensure_ascii=False))
    (args.output_dir/'scene_table_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
