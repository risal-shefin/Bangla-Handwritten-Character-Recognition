import numpy as np
from PIL import Image
from PIL import ImageOps
import cv2
import os
import csv

new_csv_filename = "test_set_e.csv"
IMG_DIR = 'images\\testing-all-corrected\\testing-e'

def write_header():

    header_lst = [[]]
    for i in range(28*28):
        header_lst[0].append('pixel'+str(i))

    with open(new_csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(header_lst)

def img_to_csv():

    for img in os.listdir(IMG_DIR):

        img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)

        img_pil = Image.fromarray(img_array)

        #invert image if needed
        img_pil = ImageOps.invert(img_pil)

        img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))

        img_array = (img_28x28.flatten())

        img_array  = img_array.reshape(-1,1).T

        # Convert float values to integer values:
        #img_array = img_array.astype(int)

        #print(img_array)

        with open(new_csv_filename, 'ab') as f:
            np.savetxt(f, img_array.astype(int), fmt="%d", delimiter=",")
            

def main():
    write_header()
    img_to_csv()

if __name__ == "__main__":
    main()