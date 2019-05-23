# Needed libirarires
import numpy as np
import h5py
import numpy as np
import h5py
import imgaug as ia
from imgaug import augmenters as iaa

def write_data(imags_matrix, targets_matrix, directory, filename):
    with h5py.File(directory + filename, 'w') as hdf:
        hdf.create_dataset('rgb',data=imags_matrix)
        hdf.create_dataset('targets',data=targets_matrix)

def shuffle_data(imgs_data, targets_data):
    s = np.arange(len(imgs_data))
    np.random.shuffle(s)
    print("Done shuffling!")
    return imgs_data[s], targets_data[s]

def augmant_data(imgs_data, number_of_augmanted_files):
    st = lambda aug: iaa.Sometimes(0.4, aug)
    oc = lambda aug: iaa.Sometimes(0.3, aug)
    rl = lambda aug: iaa.Sometimes(0.09, aug)

    seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))),                                               # blur images with a sigma between 0 and 1.5
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),     # add gaussian noise to images
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),                                # randomly remove up to X% of the pixels
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iaa.Add((-40, 40), per_channel=0.5)),                                      # adjust brightness of images (-X to Y% of original value)
        st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),                               # adjust brightness of images (X -Y % of original value)
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),                   # adjust the contrast
    ], random_order=True)
    
    augmanted_imgs = []
    for i in range(number_of_augmanted_files*32):
        augmanted_img = seq.augment_image(imgs_data[i])
        augmanted_imgs.append(augmanted_img)
    augmanted_imgs = np.array(augmanted_imgs)
    print("Done Augmanting!")
    imgs_data = np.concatenate([imgs_data, augmanted_imgs])
    return imgs_data
    
        
def Batching(x, data_dir, output_dir, number_of_files, number_of_augmanted_files): 
    total_files = int(number_of_files + number_of_augmanted_files)
    file_number = x[0] + x[1]
    index = x[0]
    
    for j in range(index, index + number_of_files):
        if(j == index):
            imgs_data = []
            targets_data = []

        if((j - index) != 0 and (j - index)%100 == 0 ):
            print("Read ", j - index, " Files", " ...")
        if((j ==  index + number_of_files - 1)):
            print("Read ", j + 1 - index, " Files", "..", " end of reading process", " ...")
            print("."*50)

        filename = data_dir + '/data_0' + str(j) + '.h5'
        with h5py.File(filename, 'r') as hdf:
            imgs = hdf.get('rgb')
            imgs = np.array(imgs[:,:,:])
            targets = hdf.get('targets')
            targets = np.array(targets)

        for i in range (0,32):

            imgs_data.append(imgs[i])
            targets_data.append(targets[i])
            
        if(j == index + number_of_files - 1):             # Calling the shuffle function! 
            imgs_data = np.array(imgs_data)
            targets_data = np.array(targets_data)
            print("Data shape before processing ..")
            print(imgs_data.shape)
            print(targets_data.shape)


            print("First shuffle ...")
            imgs_data, targets_data = shuffle_data(imgs_data, targets_data)

            print("Augmanting data")
            targets_data = np.concatenate([targets_data, targets_data[0:number_of_augmanted_files*32]])
            imgs_data = augmant_data(imgs_data, number_of_augmanted_files)

            print("Second shuffle ..")
            imgs_data, targets_data = shuffle_data(imgs_data, targets_data)

            print("Data shape after processing ..")
            print(imgs_data.shape)
            print(targets_data.shape)

            for i in range(total_files):
                output_filename = '/data_0' + str(file_number) + '.h5'
                write_data(imgs_data[i*32:((32*(i+1)))], targets_data[i*32:((32*(i+1)))], output_dir, output_filename)
                file_number = file_number + 1
            print("End of processing the batch ..")
            print("*"*70)



# Classifying data into four categories
# Follow
data_directories = '/content/drive/My Drive/AgentHuman/SeqTrain/Follow'
number_of_files = [2000, 2000, 1815]
number_of_augmanted_files = [165, 165, 155]         
                                           
for i in range(len(number_of_files)):
    print("Start of the batch")
    if(i == 0):
        x = [0, 0]
    else:
        x = [sum(number_of_files[0:i]), sum(number_of_augmanted_files[0:i])]
    Batching(x, data_directories, data_directories + '/Follow_Batch', number_of_files[i], number_of_augmanted_files[i])




# Straight
data_directories = '/content/drive/My Drive/AgentHuman/SeqTrain/Straight'
number_of_files = [1500, 1500, 1382]
number_of_augmanted_files = [700, 700, 518]            
                                          
for i in range(len(number_of_files)):
    print("Start of the batch")
    if(i == 0):
        x = [0, 0]
    else:
        x = [sum(number_of_files[0:i]), sum(number_of_augmanted_files[0:i])]
    Batching(x, data_directories, data_directories + '/Straight_Batch', number_of_files[i], number_of_augmanted_files[i])




# Right
data_directories = '/content/drive/My Drive/AgentHuman/SeqTrain/Right'
number_of_files = [1100, 1100, 1100, 1019]
number_of_augmanted_files = [810, 810, 810, 811]             
                                           
for i in range(len(number_of_files)):
    print("Start of the batch")
    if(i == 0):
        x = [0, 0]
    else:
        x = [sum(number_of_files[0:i]), sum(number_of_augmanted_files[0:i])]
    Batching(x, data_directories, data_directories + '/Right_Batch', number_of_files[i], number_of_augmanted_files[i])



# Left
data_directories = '/content/drive/My Drive/AgentHuman/SeqTrain/Left'
number_of_files = [1387, 1387, 1000]
number_of_augmanted_files = [863, 863, 800]
                                          
for i in range(len(number_of_files)):
    print("Start of the batch")
    if(i == 0):
        x = [0, 0]
    else:
        x = [sum(number_of_files[0:i]), sum(number_of_augmanted_files[0:i])]
    Batching(x, data_directories, data_directories + '/Left_Batch', number_of_files[i], number_of_augmanted_files[i])





