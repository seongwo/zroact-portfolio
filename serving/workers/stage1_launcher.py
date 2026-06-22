"""
Stage1 server launcher.
Sets sys.path for both stage1 (utils) and stage2 (serving) before importing.
"""
import argparse
import sys
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--stage1-root", required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8001)
args = parser.parse_args()

# stage2 root = two levels up from this file's directory
stage2_root = str(Path(__file__).resolve().parents[2])

sys.path.insert(0, args.stage1_root)  # for utils.*
sys.path.insert(0, stage2_root)       # for serving.*

os.environ["STAGE1_ROOT"] = args.stage1_root  # stage1_server.py module-level code reads this

from serving.workers.stage1_server import app
import uvicorn

uvicorn.run(app, host=args.host, port=args.port)
