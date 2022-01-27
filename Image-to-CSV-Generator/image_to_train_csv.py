import numpy as np
from PIL import Image
from PIL import ImageOps
import cv2
import os
import csv

label_lst = []
new_csv_filename = "train_set_e.csv"
train_label_filename = "training-e.csv"
IMG_DIR = 'images/training-e'

def write_header():

    header_lst = [['label']]
    for i in range(28*28):
        header_lst[0].append('pixel'+str(i))

    with open(new_csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(header_lst)

def gen_label():

    with open(train_label_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue

            label_lst.append(row[8])
            line_count += 1

def img_to_csv():

    img_count = 0

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

        # adding label column in front of numpy array
        img_array = np.concatenate((np.array([ label_lst[img_count] ])[:, np.newaxis], img_array), axis=1)

        #print(img_array)

        with open(new_csv_filename, 'ab') as f:
            np.savetxt(f, img_array.astype(int), fmt="%d", delimiter=",")
        
        img_count += 1

def main():
    write_header()
    gen_label()
    img_to_csv()

if __name__ == "__main__":
    main()