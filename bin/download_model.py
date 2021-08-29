import gdown
import os


if not os.path.exists('assets'):
    os.makedirs('assets')

print("\nThe model is about 2GB and will take some time to be downloaded\n", flush=True)

gdown.download(
    "https://drive.google.com/uc?id=1cdUEiMCdkFIK88zpB4P2bKnHgRc6wqWT",
    "assets/best-checkpoint.ckpt",
)
