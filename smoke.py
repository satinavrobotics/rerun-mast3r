#!/usr/bin/env python3
import os
os.environ.setdefault("RERUN_SPAWN", "false")
import rerun as rr

# Start a headless recorder
rr.init("smoke-test", spawn=False)

# Emit a couple of 2D points
rr.log("smoke/points2d", rr.Points2D([[10, 20], [30, 40]]))

# Keep the recorder alive long enough for you to connect and view
import time; time.sleep(5)
