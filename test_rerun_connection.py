#!/usr/bin/env python3
"""
Test script to verify Rerun gRPC connection between API and CLI containers.
Run this inside the master_slam_api container to test if data is being sent to the CLI viewer.
"""

import rerun as rr
import numpy as np
import time

# Initialize rerun
print("[Test] Initializing rerun...")
rr.init("connection-test", spawn=False)

# Connect to CLI server
server_addr = "master_slam_cli:9878"
print(f"[Test] Connecting to rerun server at {server_addr}...")
rr.connect_grpc(f"rerun+http://{server_addr}/proxy", flush_timeout_sec=0.1)
print(f"[Test] ✓ Connected")

# Send test data
print("[Test] Sending test data...")
for i in range(10):
    # Log a moving point
    rr.set_time_sequence("frame", i)
    rr.log("test/point", rr.Points3D([[i * 0.1, 0, 0]], colors=[[255, 0, 0]], radii=[0.05]))
    rr.log("test/text", rr.TextDocument(f"Frame {i}"))
    print(f"[Test] Logged frame {i}")
    time.sleep(0.5)

print("[Test] ✓ Sent 10 frames")
print("[Test] Waiting 3 seconds for data to be transmitted...")
time.sleep(3)
print("[Test] ✓ Done")
print("")
print("If you have a Rerun viewer connected to the CLI server, you should see:")
print("  - A red point moving along the X-axis")
print("  - Text showing 'Frame 0' through 'Frame 9'")
print("")
print("To connect a viewer from your local PC, run:")
print(f"  rerun --connect rerun+http://127.0.0.1:9878/proxy")

