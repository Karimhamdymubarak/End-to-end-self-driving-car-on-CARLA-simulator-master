# Needed libirarires
import numpy as np
import h5py

# Global variable 
data_dir = '/media/mustafa/Cloud/Graduation project/Data/AgentHuman/SeqTrain'                                  # Data directory 

high_level = ["FOLLOW", "LEFT", "RIGHT", "STRAIGHT"]                                      # High level commands
total_counter = np.zeros(4)                                                               # Used to count up images

def write_data(imags_matrix, targets_matrix, directory, filename):
    with h5py.File(directory + filename, 'w') as hdf:
        hdf.create_dataset('rgb',data=imags_matrix)
        hdf.create_dataset('targets',data=targets_matrix)
        

def classify(output_dir, command): 
    imags_counter = 0
    file_number = 0
    for _ in range(3663, 6952, 1):
        if((_ - 3663)%500 == 0 and (_ != 3663)):
            print("Done classifying 500 files ..")
        if(_ == 3663):
            imgs_data = []
            targets_data = []
        filename = data_dir + '/data_0' + str(_) + '.h5'
        if(_ == 6790):
            continue
        with h5py.File(filename, 'r') as hdf:
            imgs = hdf.get('rgb')
            imgs = np.array(imgs[:,:,:])
            targets = hdf.get('targets')
            targets = np.array(targets)
            for i in range (0,200):
                if(targets[i][10] < -1):
                	continue 
                img = imgs[i]
                target = targets[i]
                if(target[24] == command ):
                    imgs_data.append(img)
                    targets_data.append(target)
                    imags_counter = imags_counter + 1
                    if(imags_counter == 32):
                        output_filename = '/data_0' + str(file_number) + '.h5'
                        write_data(imgs_data, targets_data, output_dir, output_filename)

                        total_counter[int(command - 2)] = total_counter[int(command - 2)] + 32 
                        file_number = file_number + 1
                        imags_counter = 0
                        imgs_data = []
                        targets_data = []
        if(_ == 6951):
            print("Additional ", int(imags_counter), " images with the command ", high_level[int(command - 2)], " were neglected! ..")
            print("The total number of images associated with command ", high_level[int(command - 2)], " equals = ", int(total_counter[int(command - 2)]), " ,That is = ", int(total_counter[int(command - 2)])//32, " Files.")
            print("."*50)

            imgs_data = []
            targets_data = []


# Classifying data into five categories
output_directories = ['/media/mustafa/Cloud/Graduation project/Data/AgentHuman/SeqTrain/Follow',
                      '/media/mustafa/Cloud/Graduation project/Data/AgentHuman/SeqTrain/Left', 
                      '/media/mustafa/Cloud/Graduation project/Data/AgentHuman/SeqTrain/Right', 
                      '/media/mustafa/Cloud/Graduation project/Data/AgentHuman/SeqTrain/Straight']

commands = [2.0, 3.0, 4.0, 5.0]
for i, directory in enumerate(output_directories):
    print("Classifying the ", high_level[i], " command ...")
    classify(directory, commands[i])
