import os
import cv2
import numpy as np
import argparse
from pimage_lib import pimage as pi
import threading
import time
import matplotlib.pyplot as plt


THREADS = 24

def process_image(img_raw, dir, token):
    """ Convert raw images and stores into files
    """
    img_rgb, dolp_np, aolp_cos_np, aolp_sin_np = pi.extractNumpyArrays(img_raw)
    
    dir = '/mnt/gpu_storage/potatoseg/potato/'
    
    # Save the color image
    cv2.imwrite(os.path.join(dir+'pol_color',token+".png"), img_rgb)
    # Save the numpy arrays
    np.save(os.path.join(dir+'pol_dolp',token), dolp_np)
    np.save(os.path.join(dir+'pol_aolp_cos',token), aolp_cos_np)
    np.save(os.path.join(dir+'pol_aolp_sin',token), aolp_sin_np)

    return


def producer(filenames, img_dir, buffer, mutex, finished):
    """ Load files into the image buffer
    """
    for raw_img_name in filenames:
        token = raw_img_name[:-8] # remove _raw.png ending

        # Read image
        img_raw = cv2.imread(os.path.join(img_dir, raw_img_name), cv2.IMREAD_GRAYSCALE)

        # TODO: Add upper limit on buffern lengh
        mutex.acquire(blocking=False)
        buffer.append({'img':img_raw, 'token':token, 'dir':img_dir})
        mutex.release()

    finished.set()
    print("Finished loading all images")

    return


def consumer(buffer, mutex, finished):
    """ Process images from the image buffer
    """
    completed = False
    while True:
        if finished.is_set():
            completed = True

        mutex.acquire(blocking=False)
        buffer_len = len(buffer)
        print("Buffer length:",buffer_len, end='\r')

        if len(buffer) == 0:
            mutex.release
            if completed:
                # print("C: Finished")
                break
            else:
                # print("C: Slow producer")
                time.sleep(0.1)
        else:
            item = buffer.pop()
            # print("C: release loaded image")
            mutex.release
            process_image(item['img'], item['dir'], item['token'])

    return


def run(args):
    """ Get file list and manage threads.
    """
    img_dir = args.directory

    # Get a list of files containing the "_raw" tag
    filenames = []
    for file in os.listdir(img_dir):
        if "_raw.png" in file:
            filenames.append(file)
    filenames.sort()

    buffer = []
    mutex = threading.Lock()
    finished = threading.Event()

    prod = threading.Thread(target=producer, args=(filenames, img_dir, buffer, mutex, finished))

    # Start all threads
    prod.start()
    consumers = []
    for i in range(THREADS):
        consumers.append(threading.Thread(target=consumer, args=(buffer, mutex, finished)))
    for i in range(THREADS):
        consumers[i].start()
    print(f"Started {THREADS+1} threads")

    # Wait all threads to finish
    prod.join()
    for i in range(THREADS):
        consumers[i].join()
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract data from folder with raw images.")

    #option argument
    parser.add_argument("-d", "--directory", help="Image directory.", type=str, default="./input/small_test/")

    args = parser.parse_args()

    if os.path.isdir(args.directory) is False:
        print("Directory", args.directory,"does not exists")
        quit()

    run(args)
    print("\nFinished")
    quit()
