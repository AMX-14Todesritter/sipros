#!/usr/bin/env bash
# Run on the 5090 host from the project checkout. Outputs stay in output/evaluation.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p output/evaluation/dependencies
if [ "$(df --output=avail -B1 . | tail -n 1)" -lt 21474836480 ]; then
  echo 'Need at least 20 GiB free before restoring inputs/builds.' >&2
  exit 1
fi

if [ ! -x output/evaluation/dependencies/raxport-linux/Raxport ]; then
  if [ ! -d output/evaluation/dependencies/raxport-source/.git ]; then
    git clone https://github.com/xyz1396/Raxport.net.git output/evaluation/dependencies/raxport-source
  fi
  git -C output/evaluation/dependencies/raxport-source rev-parse HEAD > output/evaluation/dependencies/raxport-source-commit.txt
  docker run --rm -v "$PWD:/workspace/sipros" \
    -w /workspace/sipros/output/evaluation/dependencies/raxport-source \
    mcr.microsoft.com/dotnet/sdk:8.0-jammy \
    dotnet publish RaxportNetCore.csproj -c Release -r linux-x64 --self-contained true \
      -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true \
      -o /workspace/sipros/output/evaluation/dependencies/raxport-linux
fi

# One foreground container in the detached host job; a fresh GPU check precedes work.
docker compose run --rm -e NVIDIA_DRIVER_CAPABILITIES=all sipros bash -lc '
  set -e
  nvidia-smi
  mkdir -p output/evaluation/inputs output/evaluation/results/preparation
  for x in 1 2 3; do
    input="raw/Pan_062822_X${x}iso5.raw"
    if [ "$x" = 1 ]; then input="raw/raw/Pan_062822_X1iso5.raw"; fi
    dest="output/evaluation/inputs/ecoli_x${x}"
    mkdir -p "$dest"
    if [ ! -f "$dest/Pan_062822_X${x}iso5.FT2" ]; then
      output/evaluation/dependencies/raxport-linux/Raxport -f "$input" -o "$dest" -n 6 \
        > "output/evaluation/results/preparation/x${x}.log" 2>&1
    fi
  done
  python3 -B - <<"PY"
from pathlib import Path
from sys import path
path.insert(0,"evaluation/scripts")
from inventory import sha
expected={
 "output/evaluation/inputs/ecoli_x1/Pan_062822_X1iso5.FT2":"a04b10ed1c5a3862e5a7c6e48b88a61c27ae15f5e7fd52d0c3198e75e5863c8a",
 "output/evaluation/inputs/ecoli_x2/Pan_062822_X2iso5.FT2":"a78c87301400d2e5c54c7cd2780580c18ed1199eeff49d9e53e5cb773fab6409",
 "output/evaluation/inputs/ecoli_x3/Pan_062822_X3iso5.FT2":"1947349163af9b5bc586c0677f739340ddfbdc8f7fdb4f262ebfd227e3bebfb6",
 "raw/Marine_fw_3rev.fasta":"98c9b3c547d7d824d6fdde71882a8223407117cf59b61a6693e79be9107bef28",
 "experiments/Regular.cfg":"04ba2a4c120ac7207f34fce774169ced522f05e53f59398d955029705cf198e6"}
for name,digest in expected.items():
 assert sha(Path(name))==digest,"Input hash mismatch: "+name
 print("Verified:",name,flush=True)
# FT1 is not used by these searches. Conversion outputs can be regenerated.
for file in Path("output/evaluation/inputs").glob("ecoli_x*/*.FT1"):file.unlink()
PY
  if [ ! -f output/evaluation/build/gpu_multi/build_manifest.json ]; then
    python3 -B evaluation/scripts/prepare_batch_binary.py --precursor-mode multi --cuda-architectures 120
  fi
  if [ ! -f output/evaluation/inputs/scan_scaling_seed20261028/manifest.json ]; then
    python3 -B evaluation/scripts/scaling_inputs.py scans --seed 20261028 --points 140297 \
      --sources output/evaluation/inputs/ecoli_x1/Pan_062822_X1iso5.FT2 \
                output/evaluation/inputs/ecoli_x2/Pan_062822_X2iso5.FT2 \
                output/evaluation/inputs/ecoli_x3/Pan_062822_X3iso5.FT2
  fi
  exec python3 -u -B evaluation/scripts/batch_study.py \
    --id scan_5090_multi_s140297_extra4_v2 \
    --category scan_scaling --precursor-mode multi \
    --scans output/evaluation/inputs/scan_scaling_seed20261028/scans_140297.FT2 \
    --fasta raw/Marine_fw_3rev.fasta --config experiments/Regular.cfg \
    --gpu-binary output/evaluation/build/gpu_multi/bin/sipros_mvh_cuda \
    --batches 4000000 --backends cuda rt-custom --repeats 4 --warmups 0 \
    --min-device-free-gib 3 --max-host-rss-gib 55 --failure-policy stop
'
