import gdown
import tarfile
import threading
import os


def download_data():

    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    print("\nDownloading dataset...\n", flush=True)

    gdown.download(
        "https://drive.google.com/uc?id=1LtrGndz9P766BRPf-jWkRw0_gzDuVCVo",
        "dataset/keyword_categories.tar.gz",
    )


def main():
    thread = threading.Thread(target=download_data)
    thread.start()
    # wait for the data to be downloaded
    thread.join()

    # Extract dataset
    print("\nExtracting files...\n", flush=True)
    tar = tarfile.open("dataset/keyword_categories.tar.gz", "r:gz")
    tar.extractall("dataset/keyword_categories/")
    tar.close()


if __name__ == '__main__':
    main()
