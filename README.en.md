# VueParser

[简体中文](README.md) | **English**

VueParser converts Smart 3D `.vue` geometry into GLB, OBJ, and binary FBX. It supports merging meshes by component and incorporating business hierarchies and properties from MDB2 databases into GLB. The output is a visualization preview: some placements use candidate geometry or diagnostic bounding boxes, so the result should not be treated as a fully decoded solid model.

## Requirements and installation

Python 3.10 or later is required when running from source. Geometry conversion uses `numpy`, `scipy`, `trimesh`, and `mapbox-earcut`. FBX export also requires `pyassimp` and an Assimp dynamic library it can load. MDB2 property import requires `access-parser`.

```powershell
python -m pip install numpy scipy trimesh pyassimp
python -m pip install -r requirements-vue-preview.txt
python -m pip install -r requirements-vue-mdb.txt
```

You can omit `pyassimp` if you do not export FBX, and omit MDB2 dependencies if you do not import database properties. Windows executables run without a separate Python installation; see [Windows builds](#windows-builds).

## Quick start

### Windows executable

Place the single-file `VueParser.exe` in a directory and run it from PowerShell:

```powershell
.\VueParser.exe .\model.vue --format glb --merge-moniker true --unknown-bbox-wireframe false -o outputs
.\VueParser.exe --help
```

Replace `model.vue` with your input path and quote paths containing spaces. A matching `.mdb2` file in the input directory is loaded automatically. Use `--no-mdb2` to disable property import.

### Python

The command-line entry point is `vue_cli.py`. It accepts multiple existing `.vue` files and creates a subdirectory named after each input under the output directory.

```powershell
# Export GLB, OBJ, and FBX; include diagnostic bounding boxes
python vue_cli.py .\063E.vue --merge-moniker true -o outputs

# Export GLB with diagnostic reports, without unknown-geometry bounding boxes
python vue_cli.py .\063E.vue --format glb --merge-moniker true --unknown-bbox-wireframe false --reports true -o outputs\review

# Merge component meshes and select an MDB2 database explicitly
python vue_cli.py .\15300.vue --format glb --merge-moniker true --mdb2 .\15300.mdb2 -o outputs\business

# Convert multiple files
python vue_cli.py .\063E.vue .\15300.vue --format obj --merge-moniker true -o outputs\obj
```

These examples use `--merge-moniker true` to merge meshes belonging to the same component. Merging is disabled by default and must be enabled explicitly.

## Command-line options

| Option | Description | Default |
| --- | --- | --- |
| `inputs` | One or more `.vue` paths; input filenames without extensions must be distinct within a batch | Required |
| `-o`, `--output-dir` | Output root directory | `outputs` |
| `--format {all,glb,fbx,obj}` | Output format | `all` |
| `--segments N` | Circumferential divisions: multiples of 8 from 16 to 256, such as 16, 24, or 32; partial arcs use proportional divisions with a minimum of 3 | `16` |
| `--reports {true,false}` | Write CSV/JSON diagnostic reports | `false` |
| `--unknown-bbox-wireframe {true,false}` | Include magenta bounding-box lines for unknown geometry | `true` |
| `--merge-moniker {true,false}` | Merge meshes sharing a Moniker; `--merge-components` is an alias and also requires a boolean value | `false` |
| `--mdb2 PATH`, `--no-mdb2` | Select a database for GLB property import or disable import; `auto` searches for a matching `.mdb2` beside the VUE file | `auto` |
| `--glb-up-axis {y,z}` | GLB vertical axis; does not affect OBJ/FBX | `y` |
| `--fbx-mode {glb,auto}` | `glb` converts via temporary GLB; `auto` first tries the ASCII route and uses the GLB route directly when diagnostic lines are present | `glb` |
| `-h`, `--help` | Show help | — |
| `--version` | Show version | — |

The three boolean options require an explicit `true` or `false` value, case-insensitively. An explicit MDB2 path can only be used with one VUE input. If automatic database discovery finds no match, conversion continues with VUE geometry only. Temporary intermediate files for `--format fbx` are cleaned up automatically. Exit codes are `2` for argument errors, `1` for conversion errors, and `130` for user interruption.

## Output files

For input `063E.vue` and output root `outputs`, files are written to `outputs/063E/`.

| File | When generated and contents |
| --- | --- |
| `<name>.glb` | `--format glb` or `all`; includes materials and can include MDB2 hierarchy and properties |
| `<name>.obj` | `--format obj` or `all`; geometry and normals, without an MTL material file |
| `<name>_compatible.fbx` | `--format fbx` or `all`; binary FBX |
| `geometry.csv`, `gaps.csv`, `summary.json` | `--reports true`; primitive details, decoding gaps, and statistics |
| `materials.json`, `references.json`, `profile_shells.json` | `--reports true`; materials, reference boundaries, and composite-shell diagnostics |
| `business.json` | `--reports true` and successful MDB2 integration into GLB |
| `mesh_merge.json` | Both `--reports true` and `--merge-moniker true` |
| `<name>_compatible.verification.json` | `--reports true` with GLB-to-FBX conversion or FBX merge verification |

Rerunning a conversion overwrites matching outputs for the selected formats, but does not remove older files for other formats or reports. Use a new output directory when you need results corresponding exactly to the current options.

## Geometry and display

- Unknown geometry blocks, references not covered by decoded surfaces, and incomplete records may produce magenta bounding-box lines named `__UNPARSED_BBOX`. These are diagnostic placeholders. GLB uses `LINES` primitives, OBJ uses `l` records, and FBX uses two-vertex lines. Line width and FBX line visibility depend on the viewer. Use `--unknown-bbox-wireframe false` to omit them. With `--reports true`, decoding gaps remain in the diagnostic reports; `summary.json` also counts line segments.
- If disabling bounding boxes leaves no exportable geometry, conversion reports an error instead of creating an empty model.
- `--merge-moniker true` preserves materials and opacity without performing boolean unions, hole filling, or geometry simplification. GLB stores solid faces and diagnostic lines as separate primitives within a component mesh; OBJ places them in the same `o` object. FBX keeps diagnostic lines as separate nodes and merges solid meshes only.
- GLB converts source coordinates `(E, N, EL)` in metres from Z-up to Y-up by default. Use `--glb-up-axis z` to preserve source orientation. OBJ and FBX always retain source Z-up coordinates.
- MDB2 integration applies only to GLB. Object nodes can store identity and business properties in `extras.moniker` and `extras.properties`. Visibility of these fields depends on the viewer's support for glTF `extras`.
- Insulation uses the original object color with transparency. Native `Weld` objects appear as black circular lines with their original opacity. Disabling unknown bounding boxes does not remove welds. Line width depends on the viewer. Check diagnostic reports when reviewing candidate geometry or incomplete records.

## Python API

```python
from pathlib import Path
from vue_export_options import ExportOptions
from vue_preview_export import export_sample

export_sample(
    Path('063E.vue'),
    Path('outputs/063E'),
    ExportOptions(format='glb', reports=True, merge_moniker=True,
                  glb_up_axis='y', unknown_bbox_wireframe=False),
)
```

The destination passed to `export_sample(path, destination)` is the final output directory; the input name is not appended. Python defaults for `ExportOptions()` are: all formats, reports enabled, merging disabled, Y-up GLB, `auto` FBX mode, no automatic MDB2 import, and unknown bounding boxes enabled. Command-line defaults are listed above.

## Windows builds

Use a Python environment with the dependencies above, then install PyInstaller:

```powershell
python -m pip install pyinstaller

# Single-file executable
python -m PyInstaller --noconfirm --clean --distpath outputs/exe/onefile --workpath outputs/exe/build-onefile VueParser-onefile.spec
.\outputs\exe\onefile\VueParser.exe --version
.\outputs\exe\onefile\VueParser.exe .\063E.vue --format glb --merge-moniker true -o outputs\exe-check

# Directory distribution and ZIP package
.\build_exe.ps1
```

The single-file output is `outputs/exe/onefile/VueParser.exe` and can be copied on its own. It extracts dependencies to the system temporary directory at startup, which can make startup slower. The directory distribution is `outputs/exe/dist/VueParser/`; copy the entire directory, not just its EXE. Before building FBX support, confirm that `pyassimp` can load the Assimp dynamic library.

## Tests

Tests are located in `scripts/` and cover record parsing, export options, MDB2 properties, GLB orientation and merging, FBX translations, segment counts, primitive normals, welds, and surface queries. Run from the project root:

```powershell
python -m unittest discover -s scripts -p "test_*.py"
```

Sample regression tests require `063E.vue`, `15300.vue`, and their matching `.mdb2` files in the project root, plus the full export dependencies. Models and databases are not tracked in Git. Generated exports, executables, and verification reports under `outputs/` are also excluded.

## License

See [LICENSE](LICENSE).
