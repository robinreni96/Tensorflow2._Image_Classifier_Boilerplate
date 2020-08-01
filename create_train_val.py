import os 
from PIL import Image



master_source_dir = "data/source_edited/"
train_save_dir = "data/train/"
val_save_dir = "data/val/"
split_percent = 10
labels = os.listdir(master_source_dir)


def split(img_paths, save_dir, label):
    counter = 1
    mode = save_dir.strip().split("/")[1]
    save_dir = os.path.join(save_dir, label)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in img_paths:
        im = Image.open(i)
        im = im.convert('RGB')
        dsize = (224, 224) # Size Details 
        im = im.resize(dsize, Image.ANTIALIAS)
        save_path = os.path.join(save_dir, label+ "_" + mode + "_" + str(counter) + ".jpg" )
        im.save(save_path , 'JPEG', quality=120)    
        counter+=1

for i in labels:
    full_path = lambda x: os.path.join(master_source_dir,i,x)
    images = os.listdir(os.path.join(master_source_dir,i))
    images_paths = list(map(full_path,images))
    total_len = len(images_paths)
    limit = (total_len * (100 - split_percent)) // 100
    print("Label {} : {}".format(i,limit))
    train_img_paths = images_paths[:limit]
    val_img_paths = images_paths[limit:]
    
    # Saving training images
    if not os.path.exists(train_save_dir):
        os.mkdir(train_save_dir)
    split(train_img_paths, train_save_dir, i)
    
    # Saving validation images
    if not os.path.exists(val_save_dir):
        os.mkdir(val_save_dir)
    split(val_img_paths, val_save_dir, i)