import gdown

print("\nThe model is about 2GB and will take some time to be downloaded\n", flush=True)
gdown.download(
    "https://drive.google.com/uc?id=1-G0G63J3NBST2N76n0RnVKv9MttyoRJO",
    "assets/model_best_checkpoint.ckpt",
)