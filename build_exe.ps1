$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m PyInstaller --noconfirm --clean --distpath outputs/exe/dist --workpath outputs/exe/build VueParser.spec
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed' }
Copy-Item -LiteralPath README.md -Destination outputs/exe/dist/VueParser/README.md -Force
python -c "import importlib.metadata as m,json,platform;from pathlib import Path;Path('outputs/exe/dist/VueParser/BUILD_INFO.json').write_text(json.dumps({'python':platform.python_version(),'architecture':platform.machine(),'geometry_version':'v29','packages':{p:m.version(p) for p in ['numpy','scipy','trimesh','mapbox-earcut','pyassimp','pyinstaller']}},indent=2),encoding='utf-8')"
if ($LASTEXITCODE -ne 0) { throw 'Build metadata failed' }
& .\outputs\exe\dist\VueParser\VueParser.exe --version
if ($LASTEXITCODE -ne 0) { throw 'Executable startup failed' }
Compress-Archive -Path outputs/exe/dist/VueParser -DestinationPath outputs/exe/VueParser-v29-win64.zip -Force
(Get-FileHash -LiteralPath outputs/exe/VueParser-v29-win64.zip -Algorithm SHA256).Hash | Set-Content outputs/exe/VueParser-v29-win64.zip.sha256 -Encoding ASCII
Write-Host 'Ready: outputs/exe/VueParser-v29-win64.zip'
