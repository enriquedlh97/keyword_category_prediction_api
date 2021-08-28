import gdown
import os


if not os.path.exists('assets'):
    os.makedirs('assets')

print("\nThe model is about 2GB and will take some time to be downloaded\n", flush=True)

gdown.download(
    "https://drive.google.com/uc?id=1q7RRFrNkw_7tHwZX3_Gu_QnwKB75Iauh",
    "assets/best-checkpoint.ckpt",
)