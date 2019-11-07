

from PIL import Image # (pip install Pillow)
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json
import os
import sys
import random

def ensure_exists(path):
    if (not os.path.isdir(path)):
        os.mkdir(path)

"""

[{'segmentation': [[159.0, 173.5, 162.0, 169.5, 176.0, 166.5, 181.5, 161.0, 183.0, 156.5, 188.5, 154.0, 195.5, 142.0, 197.5, 135.0, 195.5, 125.0, 192.5, 121.0, 194.5, 102.0, 184.5, 82.0, 179.5, 78.0, 176.0, 72.5, 165.0, 65.5, 152.0, 64.5, 148.0, 66.5, 142.5, 73.0, 139.5, 89.0, 135.5, 97.0, 136.5, 104.0, 128.5, 114.0, 127.5, 120.0, 130.5, 126.0, 136.5, 132.0, 136.5, 144.0, 142.5, 152.0, 142.5, 156.0, 146.5, 165.0, 151.0, 171.5, 154.0, 173.5, 159.0, 173.5], [115.0, 95.5, 118.5, 94.0, 119.5, 88.0, 115.0, 82.5, 111.5, 86.0, 111.5, 89.0, 114.5, 90.0, 115.0, 95.5]], 'iscrowd': 0, 'image_id': 30546, 'category_id': 1, 'id': 30546, 'bbox': [111.5, 64.5, 86.0, 109.0], 'area': 5303.75}, None, {'segmentation': [[150.0, 177.5, 154.0, 176.5, 157.0, 173.5, 163.0, 173.5, 174.0, 169.5, 187.5, 157.0, 194.5, 147.0, 195.5, 141.0, 199.5, 134.0, 197.5, 120.0, 201.5, 114.0, 201.5, 111.0, 195.5, 92.0, 188.5, 81.0, 178.0, 70.5, 167.0, 62.5, 162.0, 60.5, 145.0, 58.5, 126.0, 62.5, 111.0, 62.5, 106.0, 65.5, 99.0, 66.5, 69.0, 81.5, 60.5, 96.0, 60.5, 99.0, 56.5, 107.0, 56.5, 112.0, 60.5, 118.0, 56.5, 122.0, 58.5, 128.0, 58.5, 136.0, 60.5, 139.0, 60.5, 145.0, 66.5, 154.0, 69.0, 156.5, 81.0, 161.5, 83.5, 164.0, 83.0, 168.5, 89.0, 169.5, 93.0, 166.5, 94.5, 168.0, 93.5, 170.0, 97.0, 172.5, 102.0, 173.5, 104.0, 169.5, 106.0, 169.5, 113.0, 173.5, 117.0, 173.5, 119.0, 176.5, 123.0, 172.5, 131.0, 175.5, 150.0, 177.5]], 'iscrowd': 0, 'image_id': 30548, 'category_id': 1, 'id': 30548, 'bbox': [56.5, 58.5, 145.0, 119.0], 'area': 13644.75}]
[{'segmentation': [[133.0, 180.5, 137.0, 178.5, 153.0, 177.5, 156.0, 175.5, 169.0, 172.5, 177.0, 167.5, 184.0, 158.5, 187.5, 157.0, 190.5, 150.0, 194.5, 146.0, 194.5, 138.0, 197.5, 133.0, 195.5, 130.0, 195.5, 112.0, 199.5, 100.0, 197.5, 94.0, 191.5, 89.0, 188.5, 84.0, 188.5, 81.0, 178.0, 69.5, 174.0, 68.5, 167.0, 62.5, 163.0, 63.5, 156.0, 59.5, 150.0, 58.5, 118.0, 59.5, 96.0, 66.5, 90.0, 66.5, 84.0, 71.5, 73.0, 76.5, 63.5, 86.0, 55.5, 98.0, 55.5, 109.0, 57.5, 118.0, 54.5, 122.0, 55.5, 126.0, 53.5, 129.0, 53.5, 135.0, 57.5, 148.0, 62.5, 157.0, 65.0, 157.5, 76.0, 167.5, 88.0, 170.5, 92.0, 175.5, 98.0, 174.5, 113.0, 179.5, 133.0, 180.5]], 'iscrowd': 0, 'image_id': 30712, 'category_id': 1, 'id': 30712, 'bbox': [53.5, 58.5, 146.0, 122.0], 'area': 14328.25}, None, {'segmentation': [[126.0, 146.5, 146.0, 142.5, 147.0, 144.5, 149.0, 144.5, 161.5, 133.0, 165.5, 125.0, 164.5, 116.0, 158.0, 115.5, 156.5, 114.0, 157.5, 110.0, 155.5, 109.0, 156.5, 104.0, 154.0, 98.5, 150.0, 98.5, 147.0, 102.5, 142.0, 97.5, 132.0, 97.5, 127.0, 99.5, 123.0, 103.5, 122.5, 102.0, 128.5, 94.0, 127.0, 92.5, 117.0, 93.5, 116.0, 95.5, 112.0, 95.5, 109.0, 98.5, 105.0, 98.5, 102.0, 102.5, 98.5, 104.0, 93.5, 111.0, 91.5, 120.0, 94.5, 123.0, 93.5, 133.0, 97.0, 135.5, 99.0, 133.5, 111.0, 144.5, 115.0, 144.5, 116.0, 146.5, 126.0, 146.5]], 'iscrowd': 0, 'image_id': 30714, 'category_id': 1, 'id': 30714, 'bbox': [91.5, 92.5, 74.0, 54.0], 'area': 2911.0}]

printing None sometimes for annotations

"""

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))
            # print(mask_image.getpixel((x,y)))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = '1' # str(pixel) label entire tumor as 1
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        # print()
        # print(contour)
        # Make a polygon and simplify it
        poly = Polygon(contour)
        # print(poly)
        poly = poly.simplify(1.0, preserve_topology=False)
        # print(poly)
        if poly.is_empty:
            continue
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    if(len(polygons) == 0):
        return None

    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def create_coco_instances(input_dir, output_dir, file_names, dataset_name):
    annotations_dir = os.path.join(output_dir, 'annotations')
    
    ensure_exists(annotations_dir)
    
    annotation_file = open(os.path.join(annotations_dir, 'instances_'+ dataset_name+'.json'), 'a+')
    dataset_dir = os.path.join(output_dir, dataset_name)
    
    ensure_exists(dataset_dir)

    data = {}
    data["images"] = create_images(input_dir, output_dir, file_names, dataset_name)
    print("done creating images")
    data["annotations"] = create_annotations(data["images"])
    data["categories"] = [{"name":"tumor", "id":1}]


    annotation_file.write(json.dumps(data))
    annotation_file.close

def create_images(brats_dir, file_names, dataset_name):
    """
    {[

    "height": 427,
        "width": 640,
        "id": 1,
    "file_name" : path\path.jpg}
    ]
    """
    # These ids will be automatically increased as we go
    image_id = 1

    # Create the annotations
    images = []
    for name in file_names:
        image = {}
        img_path = os.path.join(brats_dir,'images', name)
        i = Image.open(img_path)
        width, height = i.size
        
        image["id"] = image_id
        image["height"] = height
        image["width"] = width
        image["file_name"] = img_path

        images.append(image)
        image_id += 1

    return images

def create_annotations(images):
    
    # Define which colors match which categories in the images
    category_ids = {'1': 1} # (255, 255, 255) = tumor_id = 1
    # category_ids = {'1': 1, '2': 1, '3': 1, '4': 1} # (255, 255, 255) = tumor_id = 1
    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1

    # Create the annotations
    annotations = []
    for i in images:
        mask_image = Image.open(i["file_name"].replace('images','labels'))
        sub_masks = create_sub_masks(mask_image)
        
        # print(mask_image)
        # print(sub_masks.items())
        for color, sub_mask in sub_masks.items():
            category_id = category_ids[color]
            annotation = create_sub_mask_annotation(sub_mask, i["id"], category_id, annotation_id, is_crowd)

            if annotation is None:
                continue

            annotations.append(annotation)
            annotation_id += 1
            if(annotation_id % 1000 == 0):
                print(str(annotation_id))
                print(str(annotation))
                print(str())

    return annotations


def main():

    output_dir = sys.argv[1] #'/home/kyle/datasets/brats_voc'
    brats_coco_dir = sys.argv[2] #'/home/kyle/datasets/brats_voc'

    for f in os.listdir(os.path.join(output_dir)):
        for g in os.listdir(os.path.join(output_dir, f)):
            input_dir = os.path.join(output_dir, f, g, "images")
            files = os.listdir(input_dir)
            # print(f)
            # print(g)
            # # print(files)


            # automatically shuffles and splits trainval data 80% / 20% 
            random.Random(1).shuffle(files)
            trainval_split = int(0.8*len(files))
            train_names = files[:trainval_split]
            val_names = files[trainval_split:]

            create_coco_instances(input_dir, output_dir, train_names, f+"_"+g+"_"+"train2017")
            create_coco_instances(input_dir, output_dir, val_names, f+"_"+g+"_"+"val2017")


if __name__ == "__main__":
    main()



"""
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json, ast
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
from scipy.io import savemat
import os
import cv2
from PIL import Image, ImageDraw
from matplotlib.patches import Polygon
from pascal_voc_writer import Writer
from shutil import copyfile
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def visual_coco(json_path, img_path1, mask_path):

    #dataDir = img_path
    annDir = json_path
    maskDir = mask_path
    #os.mkdir(maskDir)
    for filename in os.listdir(annDir):
        print(filename)
        # if os.path.exists(os.path.join(maskDir, filename.replace('.json', '.png'))):
        #     continue
        statinfo = os.stat(os.path.join(annDir, filename))
        if(statinfo.st_size ==0):
            continue
       # filename = 'pic_5343.json'
        if os.path.exists(os.path.join(img_path1, filename.replace('.json','.jpg'))):
            dataDir = img_path1

        else:
            continue
        if os.path.exists(os.path.join(maskDir, filename.replace('.json', '_mask.png'))):
            continue
        annFile = os.path.join(annDir, filename)

        dataset = json.load(open(annFile))
        anns = dataset['shapes']
        polygons = []
        color =  []
        c = [1, 1, 1]
        poly = Polygon(anns[0]['points'])

        I = cv2.imread(os.path.join(dataDir, filename.replace('.json','.jpg')))

        t = np.zeros((I.shape[0], I.shape[1]))
        width = I.shape[1]
        height = I.shape[0]
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(totuple(poly.xy), outline=255, fill=255)
        #plt.imshow(img)
        #mask = np.array(img)
        img.save(os.path.join(maskDir, filename.replace('.json','.png')))
        Img = Image.fromarray(I)
        ImageDraw.Draw(Img).polygon(totuple(poly.xy), outline=(255, 0, 0), fill=(0,0,255))

        Img.save(os.path.join(maskDir, filename.replace('.json', '_mask.png')))

        i1 = I[:, :, 0]
        i2 = I[:, :, 1]
        i3 = I[:, :, 2]
        img1 = Image.fromarray(i1)
        ImageDraw.Draw(img1).polygon(totuple(poly.xy), outline=255, fill=255)
        i1 = np.asarray(img1)

        img2 = Image.fromarray(i2)
        ImageDraw.Draw(img2).polygon(totuple(poly.xy), outline=255, fill=255)
        i2 = np.asarray(img2)

        I2 = I
        I2[:, :, 0] = i1
        I2[:, :, 1] = i2

        Img2 = Image.fromarray(I2)
        Img2.save(os.path.join(maskDir, filename.replace('.json', '_blend.png')))

def coco2voc(json_path, img_path, path_voc):
    imgDir = img_path
    maskDir = json_path

    voc_annDir = path_voc + 'Annotations/' #xml
    voc_imageset = path_voc + 'ImageSets/'
    if not os.path.exists(voc_imageset):
        os.mkdir(voc_imageset)
    voc_segImg = path_voc + 'SegImages/'
    if not os.path.exists(voc_segImg):
        os.mkdir(voc_segImg)
    voc_imageset = path_voc + 'ImageSets/Segmentation/'
    voc_images = path_voc + 'JPEGImages/'
    if not os.path.exists(voc_annDir):
        os.mkdir(voc_annDir)
    if not os.path.exists(voc_imageset):
        os.mkdir(voc_imageset)
    if not os.path.exists(voc_images):
        os.mkdir(voc_images)

    # Annotation & JPEGImages
    for filename in os.listdir(imgDir):
        print(filename)
        maskname = filename.replace('.jpg', '.png')
        xmlname = filename.replace('.jpg', '.xml')
        copyfile(os.path.join(maskDir, maskname), os.path.join(voc_segImg, maskname))
        I_Mask = cv2.imread(os.path.join(maskDir, maskname))
        if (os.path.exists(os.path.join(voc_annDir, xmlname))):
            continue
        if not(os.path.exists(os.path.join(maskDir, maskname))):
            continue
        I_img = cv2.imread(os.path.join(imgDir, filename))
        a = np.where(I_Mask>0)
        bbox = [np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1]) + 1]
        # create writer object for .xml
        writer = Writer(os.path.join(imgDir, filename), I_img.shape[1], I_img.shape[0])
        # add object
        writer.addObject('person', np.min(a[0]), np.min(a[1])+1, np.max(a[0]), np.max(a[1]) + 1)
        writer.save(os.path.join(voc_annDir, xmlname))
        copyfile(os.path.join(imgDir, filename), os.path.join(voc_images, filename))

    f_train = open(os.path.join(voc_imageset, 'train.txt'), 'a+')
    f_trainval = open(os.path.join(voc_imageset, 'trainval.txt'), 'a+')
    f_test = open(os.path.join(voc_imageset, 'test.txt'), 'a+')
    f_val = open(os.path.join(voc_imageset, 'val.txt'), 'a+')
    i = 0
    for filename in os.listdir(imgDir):
        if i % 5 == 1:
            f_test.write(filename + '\n')
        elif i % 5 == 2:
            f_trainval.write(filename + '\n')
            f_val.write(filename + '\n')
        else:
            f_trainval.write(filename + '\n')
            f_train.write(filename + '\n')
        i = i + 1
def main():

    # path_coco = '/mnt/1FDA52311FC212D0/Matting/data/people_July_12_18/'
    # path_coco = '/mnt/1FDA52311FC212D0/Matting/data/people_June_28/'
    # visual_coco(path_coco)

    #path_coco = '/mnt/1FDA52311FC212D0/Matting/data/people_July_12_18/'
    #path_coco = '/mnt/1FDA52311FC212D0/Matting/data/people_June_28/'
    #json_path = '//mnt/1FDA52311FC212D0/Matting/data/people_July_23/mask'
    # img_path1 = '/mnt/1FDA52311FC212D0/Matting/data/people_July_23/people'
    json_path = './Aug_21/jsons/'
    img_path1 = './Aug_21/imgs/'
    # img_path2 = '/mnt/1FDA52311FC212D0/Matting/data/people_July_23/people'
    mask_path = './Aug_21/blend'
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    # path_voc = '/mnt/1FDA52311FC212D0/Matting/data/voc_July_23/'
    # if not os.path.exists(path_voc):
    #     os.mkdir(path_voc)
    visual_coco(json_path, img_path1, mask_path)
    # coco2voc(mask_path, img_path1, path_voc)
if __name__ == '__main__':
    main()"""

