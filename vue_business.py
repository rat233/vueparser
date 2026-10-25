"""MDB2 semantics and compact, identity-preserving GLB business trees."""
from collections import Counter
import hashlib
from pathlib import Path


def canonical(value):
    return str(value or '').strip().replace('@a=', '', 1).replace('!!', '-', 1).replace('##', '-', 1)


def present(value):
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def load_business(database, records, native, source):
    from vue_mdb_export import read_business_database
    rows, columns, report = read_business_database(database)
    if any(report[k] for k in ('duplicate_linkage_indexes', 'duplicate_label_name_indexes',
                              'duplicate_label_value_indexes', 'missing_label_linkage_foreign_keys',
                              'missing_label_name_foreign_keys', 'missing_label_value_foreign_keys')):
        raise ValueError('MDB2 has ambiguous or broken table references; inspect with vue_mdb_export.py')
    objects = {}
    mdb_ids = set()
    for row, long_rows in rows:
        identity = canonical(row['moniker'])
        if not identity:
            raise ValueError('MDB2 linkage has no Moniker')
        mdb_ids.add(identity)
        props = {k: row[k] for k in ['file_name', *columns] if present(row.get(k))}
        for key in ('SP3d Moniker', 'OID'):
            if key in props and canonical(props[key]) == identity:
                del props[key]
        if identity not in objects:
            objects[identity] = dict(properties=props, in_vue=False)
        else:
            target = objects[identity]['properties']
            for key, value in props.items():
                if key not in target:
                    target[key] = value
                elif target[key] != value:
                    previous = target[key] if isinstance(target[key], list) else [target[key]]
                    if value not in previous:
                        target[key] = previous + [value]
        for row in long_rows:
            extra = {}
            # Both samples use one shared empty-text/0.0 value row as the empty sentinel.
            # Numeric values paired with text are Access search caches, not extra properties.
            if not present(row['label_value']) and row['label_value_numeric'] not in (None, 0, 0.0):
                extra['numeric'] = row['label_value_numeric']
            if row['extended_label']:
                extra['extended'] = row['extended_label']
            if extra:
                extra['label'] = row['label_name']
                target = objects[canonical(row['moniker'])].setdefault('label_details', [])
                if extra not in target:
                    target.append(extra)
    vue_ids = set()
    for record in records:
        identity = canonical(record['moniker']); vue_ids.add(identity)
        obj = objects.setdefault(identity, dict(properties={}, in_vue=False))
        obj['in_vue'] = True
        obj['native_type'] = native.get(record['record_offset'], {}).get('type_name', '元件')
    report.update(database=str(database), vue_objects=len(vue_ids), mdb_objects=len(mdb_ids),
                  matched=len(vue_ids & mdb_ids), vue_only=sorted(vue_ids-mdb_ids),
                  mdb_only=sorted(mdb_ids-vue_ids))
    return dict(objects=objects, report=report, source=source, mdb_ids=mdb_ids)


def apply_business_tree(tree, business):
    """Replace identity scaffolding after mesh merging; keep mesh transforms intact."""
    objects = business['objects']
    old_nodes = tree['nodes']
    root = next(n for n in old_nodes if n.get('name') == 'VUE_Z_UP_METRES')
    nodes = [{k: v for k, v in root.items() if k != 'children'}]
    nodes[0]['name'] = business['source']
    directories = {(): 0}
    def attach(node, parent):
        index = len(nodes); nodes.append(node)
        nodes[parent].setdefault('children', []).append(index)
        return index
    def directory(parts):
        for count in range(1, len(parts)+1):
            prefix = parts[:count]
            if prefix not in directories:
                candidates = hosts.get((prefix[:-1], prefix[-1]), [])
                if len(candidates) == 1:
                    directories[prefix] = place(candidates[0])
                else:
                    directories[prefix] = attach({'name': prefix[-1]}, directories[prefix[:-1]])
        return directories[parts]
    prepared = {}
    for identity, obj in sorted(objects.items()):
        props = obj['properties']
        raw_path = props.get('System Path', '')
        path = raw_path if isinstance(raw_path, str) else ''
        parts = tuple(part.strip() for part in path.split('\\') if part.strip())
        if not parts:
            parts = ('未分类' if identity in business['mdb_ids'] else '未匹配属性',)
        raw_name = props.get('Name', '')
        name = raw_name.strip() if isinstance(raw_name, str) else ''
        if not name:
            for key in ('Equipment Name', 'Tag', 'Type'):
                candidate = props.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    name = candidate.strip(); break
            name = (name or obj.get('native_type') or '元件') + '_' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        prepared[identity] = (parts, name)
    names = Counter(prepared.values())
    business['report'].update(duplicate_path_name_groups=sum(n > 1 for n in names.values()),
        missing_names=sum(not present(o['properties'].get('Name')) for o in objects.values()),
        missing_paths=sum(not present(o['properties'].get('System Path')) for o in objects.values()),
        conflicting_properties=[dict(moniker=i, field=k) for i,o in objects.items()
            for k,v in o['properties'].items() if isinstance(v,list)])
    # 子对象挂载：对象 B 的 System Path 末段等于对象 A 的 Name、且去掉末段后
    # 等于 A 的 System Path 时（设备管嘴 15300-V-130A-M1 → 设备 15300-V-130A），
    # B 直接挂到 A 的对象节点下，而不是新建一个同名推断目录节点。
    hosts = {}
    for identity, (parts, name) in prepared.items():
        hosts.setdefault((parts, name), []).append(identity)
    # Distinct identities (e.g. body and insulation) can share a business
    # address. Preserve them under one neutral assembly instead of guessing
    # which one owns the nozzles. Missing paths are not assembly evidence.
    shared_hosts = {key for key, identities in hosts.items() if len(identities) > 1
                    and all(present(objects[i]['properties'].get('System Path'))
                            for i in identities)}
    parent_object = {}
    ambiguous_hosts = []
    for identity, (parts, name) in prepared.items():
        if parts:
            candidates = [c for c in hosts.get((parts[:-1], parts[-1]), [])
                          if c != identity]
            if len(candidates) == 1:
                parent_object[identity] = candidates[0]
            elif len(candidates) > 1:
                # Keep the ambiguity in the report; the shared directory owns
                # both candidates and their attachments without merging IDs.
                ambiguous_hosts.append(identity)
    owner_nodes = {}
    used_names = set()
    pending = {}
    for identity, (parts, name) in prepared.items():
        obj = objects[identity]; props = dict(obj['properties'])
        if names[(parts, name)] > 1:
            name += '_' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        while (parts, name) in used_names:
            name += '_'
        used_names.add((parts, name))
        if props.get('Name') == name:
            props.pop('Name')
        if props.get('System Path') == '\\'.join(parts):
            props.pop('System Path')
        extras = dict(moniker=identity, properties=props)
        if obj.get('label_details'):
            extras['label_details'] = obj['label_details']
        if not obj['in_vue']:
            extras['status'] = 'mdb_only'
        elif identity not in business['mdb_ids']:
            extras['status'] = 'vue_only'
        pending[identity] = dict(name=name, extras=extras)
    placing = set()
    def place(identity):
        if identity in owner_nodes:
            return owner_nodes[identity]
        if identity in placing:  # 防御环形 System Path
            return directory(prepared[identity][0])
        placing.add(identity)
        parts, name = prepared[identity]
        parent = directory(parts + (name,)) if (parts, name) in shared_hosts else directory(parts)
        placing.discard(identity)
        owner_nodes[identity] = attach(pending[identity], parent)
        return owner_nodes[identity]
    for identity in prepared:
        place(identity)
    counts = Counter()
    for node in old_nodes:
        if 'mesh' not in node:
            continue
        mesh = tree['meshes'][node['mesh']]
        identity = canonical(mesh.get('extras', {}).get('moniker'))
        if identity not in owner_nodes:
            raise ValueError('Mesh has no business identity: ' + str(node.get('name')))
        counts[identity] += 1
        label = '几何_' + str(counts[identity])
        new_node = {k: v for k, v in node.items() if k not in ('name', 'children', 'extras')}
        new_node['name'] = label
        attach(new_node, owner_nodes[identity])
        mesh['name'] = label
        mesh.pop('extras', None)
    for index, material in enumerate(tree.get('materials', [])):
        material['name'] = '材质_' + str(index+1)
    tree['nodes'] = nodes
    tree['scenes'] = [{'nodes': [0]}]
    tree['scene'] = 0
    tree.setdefault('extras', {})['vue_business'] = dict(version=1, source=business['source'],
        hierarchy='System Path + Name; inferred directory nodes', properties='node.extras.properties')
    business['report'].update(object_nodes=len(owner_nodes),
                              directory_nodes=sum('moniker' not in nodes[i].get('extras', {})
                                                  for i in set(directories.values()) if i != 0),
                              geometry_nodes=sum(counts.values()), objects_with_geometry=len(counts),
                              mounted_to_parent=len(parent_object),
                              shared_object_groups=len(shared_hosts),
                              ambiguous_mounts=len(ambiguous_hosts))


def resolve_database(source, setting):
    if setting is None:
        return None
    if setting == 'auto':
        path = source.with_suffix('.mdb2')
        return path if path.is_file() else None
    path = Path(setting)
    if not path.is_file():
        raise ValueError('MDB2 file not found: ' + str(path))
    return path
