# VueParser

**简体中文** | [English](README.en.md)

VueParser 将 Smart 3D 的 `.vue` 几何数据转换为 GLB、OBJ 和二进制 FBX。支持按构件合并网格，并将 MDB2 数据库中的业务树和属性整合到 GLB。转换结果是可视化预览：部分布局仍采用候选几何或诊断包围盒，不能将模型视为完整解析的实体。

## 环境与依赖

需要 Python 3.10 或更高版本。模型转换使用 `numpy`、`scipy`、`trimesh` 和 `mapbox-earcut`；FBX 转换还需要 `pyassimp` 及可由它加载的 Assimp 动态库。读取 MDB2 属性需要 `access-parser`。

```powershell
python -m pip install numpy scipy trimesh pyassimp
python -m pip install -r requirements-vue-preview.txt
python -m pip install -r requirements-vue-mdb.txt
```

只处理 VUE 几何且不导出 FBX 时，可以不安装 `pyassimp` 和 MDB2 依赖。Windows 可使用 `build_exe.ps1` 构建无需 Python 环境的目录式可执行程序，入口为 `outputs/exe/dist/VueParser/VueParser.exe`；构建还需要 PyInstaller。

## 快速开始

### 使用 Windows EXE

将单文件 `VueParser.exe` 放入任意目录，在 PowerShell 中运行：

```powershell
.\VueParser.exe .\model.vue --format glb --merge-moniker true --unknown-bbox-wireframe false -o outputs
.\VueParser.exe --help
```

将 `model.vue` 替换为实际输入路径；路径包含空格时请使用引号。EXE 无需安装 Python。同目录下的同名 `.mdb2` 会自动加载；使用 `--no-mdb2` 可关闭属性整合。

### 使用 Python

主要命令行入口是 `vue_cli.py`，支持一次输入多个已存在的 `.vue` 文件。每个文件在输出根目录下建立同名子目录。

```powershell
# 默认导出 GLB、OBJ、FBX；保留未知几何体的紫红色包围盒线框
python vue_cli.py .\063E.vue --merge-moniker true -o outputs

# 只导出 GLB，关闭未知几何体包围盒线框，并生成诊断报告
python vue_cli.py .\063E.vue --format glb --merge-moniker true --unknown-bbox-wireframe false --reports true -o outputs\review

# 合并同一 Moniker 的网格，并显式指定 MDB2 属性数据库
python vue_cli.py .\15300.vue --format glb --merge-moniker true --mdb2 .\15300.mdb2 -o outputs\business

# 批量转换
python vue_cli.py .\063E.vue .\15300.vue --format obj --merge-moniker true -o outputs\obj
```

示例使用 `--merge-moniker true` 合并同一构件的网格。命令行默认不合并，需显式设置。

## 命令行参数

| 参数 | 作用 | 默认值 |
| --- | --- | --- |
| `inputs` | 一个或多个 `.vue` 文件路径；同一批文件的文件名（不含扩展名）不能重复 | 必填 |
| `-o`, `--output-dir` | 输出根目录 | `outputs` |
| `--format {all,glb,fbx,obj}` | 选择模型格式 | `all` |
| `--segments N` | 圆周分段数，16～256 内的 8 的倍数，例如 16、24、32；局部圆弧按角度比例分段，最少 3 段 | `16` |
| `--reports {true,false}` | 生成或关闭 CSV/JSON 诊断报告 | `false` |
| `--unknown-bbox-wireframe {true,false}` | 导出或省略未知几何体的紫红色包围盒线框 | `true` |
| `--merge-moniker {true,false}` | 将同一 Moniker 下的网格合并；`--merge-components` 是别名，也需要传入 `true` 或 `false` | `false` |
| `--mdb2 PATH`, `--no-mdb2` | GLB 使用指定 MDB2 数据库，或关闭属性整合；`auto` 查找 VUE 同目录下的同名 `.mdb2` | `auto` |
| `--glb-up-axis {y,z}` | GLB 的竖直轴；不影响 OBJ/FBX | `y` |
| `--fbx-mode {glb,auto}` | `glb` 经临时 GLB 转 FBX；`auto` 优先尝试 ASCII 路线，遇到诊断线段时直接使用 GLB 路线 | `glb` |
| `-h`, `--help` | 显示帮助 | — |
| `--version` | 显示版本 | — |

上述三个布尔参数都需要显式填写 `true` 或 `false`，单独写参数名会报错；值不区分大小写。显式指定 `--mdb2 PATH` 时只能输入一个 VUE 文件。未发现自动匹配的 MDB2 时，程序会提示并继续导出 VUE 几何。`--format fbx` 使用的中间文件会自动清理。转换失败返回非零退出码；参数错误为 `2`，转换异常为 `1`，用户中断为 `130`。

## 输出文件

对于输入 `063E.vue` 和输出根目录 `outputs`，结果保存在 `outputs/063E/`。

| 文件 | 生成条件与内容 |
| --- | --- |
| `<文件名>.glb` | `--format glb` 或 `all`；包含材质，可整合 MDB2 业务树与属性 |
| `<文件名>.obj` | `--format obj` 或 `all`；只含几何与法线，不生成 MTL 材质文件 |
| `<文件名>_compatible.fbx` | `--format fbx` 或 `all`；二进制 FBX |
| `geometry.csv`, `gaps.csv`, `summary.json` | `--reports true`；图元明细、解析缺口及统计 |
| `materials.json`, `references.json`, `profile_shells.json` | `--reports true`；材质、参考边界与组合壳体诊断 |
| `business.json` | `--reports true` 且 GLB 成功整合 MDB2 |
| `mesh_merge.json` | 同时启用 `--reports true` 和 `--merge-moniker true` |
| `<文件名>_compatible.verification.json` | `--reports true` 且执行 GLB→FBX 转换或 FBX 合并核验 |

重复运行会覆盖本次选择的同名结果，但不会清理输出目录中先前生成的其他格式和报告。需要一组完全对应当前参数的文件时，请使用新输出目录。

## 模型与诊断规则

- 未识别的几何块、未能由已解析表面覆盖的引用，以及无法完整解析的记录，可生成名为 `__UNPARSED_BBOX` 的紫红色包围盒线段。GLB 使用 `LINES` 图元、OBJ 使用 `l` 记录，FBX 使用两顶点线段；线宽及 FBX 线段的显示支持由查看器决定。这些线段是诊断占位，不是源模型实体。默认导出；使用 `--unknown-bbox-wireframe false` 可从模型中移除。若启用 `--reports true`，解析缺口仍保留在 `geometry.csv`、`gaps.csv` 和 `summary.json` 中，后者单独统计 `line_segments`。
- 如果关闭包围盒后没有任何可导出的网格，转换会报告错误，不生成空模型。
- `--merge-moniker true` 保留分片的材质及透明度，不进行布尔并集、补洞或几何简化。GLB 将同一 Moniker 的实体三角面和诊断线段放在同一 mesh 的不同图元中；OBJ 将它们放在同一个 `o` 对象中。FBX 保留诊断线为独立节点，仅合并实体网格。
- GLB 默认将 VUE 的米制 `(E, N, EL)`、Z 轴向上坐标转换为 Y 轴向上；`--glb-up-axis z` 保留源朝向。OBJ 和 FBX 始终使用源 Z 轴向上坐标。
- MDB2 只整合到 GLB。GLB 对象节点可在 `extras.moniker` 和 `extras.properties` 中保存身份及业务属性；查看器是否展示这些字段取决于其对 glTF `extras` 的支持。
- 保温层使用原对象颜色和透明显示；原生 `Weld` 类型使用黑色圆周线预览，保留原透明度；关闭未知包围盒不会移除焊缝。线宽由查看器决定。候选几何和未解析记录应结合诊断报告核查。

## Python 接口

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

直接调用 `export_sample(path, destination)` 时，`destination` 是最终文件目录，不再按输入名追加子目录。`ExportOptions()` 的 Python 默认值为：导出全部格式、生成报告、不合并、GLB 转换为 Y 轴向上、FBX 使用 `auto` 路线、不自动加载 MDB2，并保留未知几何体包围盒。命令行默认值以上表为准。

## Windows 打包

使用已安装上述依赖的 Python 环境，并安装 PyInstaller：

```powershell
python -m pip install pyinstaller

# 单文件 EXE，运行时无需另装 Python
python -m PyInstaller --noconfirm --clean --distpath outputs/exe/onefile --workpath outputs/exe/build-onefile VueParser-onefile.spec
.\outputs\exe\onefile\VueParser.exe --version
.\outputs\exe\onefile\VueParser.exe .\063E.vue --format glb --merge-moniker true -o outputs\exe-check

# 目录式 EXE 和 ZIP 发布包
.\build_exe.ps1
```

单文件产物为 `outputs/exe/onefile/VueParser.exe`，可单独复制使用；启动时会将依赖解压到系统临时目录，因此首次启动可能较慢。目录式产物为 `outputs/exe/dist/VueParser/`，需要复制整个目录，不能只复制其中的 EXE。FBX 打包前应确认 `pyassimp` 能正常加载 Assimp 动态库。

## 测试

测试统一位于 `scripts/`，覆盖记录解析、导出选项、MDB2 属性、GLB 朝向及合并、FBX 位移、分段数、图元法线、焊缝和曲面查询。从项目根目录执行：

```powershell
python -m unittest discover -s scripts -p "test_*.py"
```

样本回归需要根目录中的 `063E.vue`、`15300.vue` 及同名 `.mdb2` 文件，并安装完整导出依赖；这些模型和数据库不纳入 Git。`outputs/` 中的导出结果、打包产物和验证报告也不纳入 Git。

## 许可证

参见 [LICENSE](LICENSE)。
