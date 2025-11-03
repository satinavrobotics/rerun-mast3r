docker exec -it master_slam_api bash
cd rerun_mast3r
conda activate mast3r-slam

# On local pc:
# rerun --connect rerun+http://0.0.0.0:9878/proxy

# CLI usage example:
python /workspace/rerun_mast3r/sati_master_slam.py \
  --dataset /workspace/dataset/rgb_no1vcF_17_0 \
  --config /workspace/rerun_mast3r/config/calib.yaml \
  --save-as stanford_out \
  --img-size 512 \
  --all-frames \
  --rr-config.headless \
  --rerun-server-addr master_slam_cli:9878 \
  --full-slam \
  --conf-thresh 0.0
  