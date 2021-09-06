import gdown
import os


if not os.path.exists('assets'):
    os.makedirs('assets')

if not os.path.exists('assets/bert'):
    os.makedirs('assets/bert')

print("\nThe model is about 2GB and will take some time to be downloaded\n", flush=True)

gdown.download(
    "https://drive.google.com/uc?id=1fPqgyH8PFpur_r3oOPLfF3IEBTd4Zl_e",
    "assets/bert/best-checkpoint.ckpt",
)
