# Names of Group 1 participants:
#   Isabella Junker Hacke, Gustav Nicolay Meilby Nobel, Henrik Schwarz,
#   Thobias Moldrup Sahi Aggerholm, Rasmus Boegeholt Vandkilde.

# Task 1 - Dataset: Get current directory and create 3 folders; training, test and validation.
import os       # Imports functions that allows interaction with the local OS.
import random   # Randomizing samples used in the dataset division, giving us unique objects each run.
import glob     # Loads filepath for each file/folder.
import shutil   # Lib used for copying/removing files/folders.

# Creates folder for distributed datasets.
if not os.path.exists('./dataset_catdog/dataset'): # if none exists.
    os.makedirs('./dataset_catdog/dataset')
data_path = "./dataset_catdog/dataset"


#Folders to be created
datasets = ["training", "test", "validation"]
datatypes = ["cats", "dogs"]

'''
Datastructure created in the next function:
--datasets_catdog
    --dataset
        --training
            --cats
            --dogs
        --test
            --cats
            --dogs
        --validation
            --cats
            --dogs
'''

# Create structure detailed above
for dataset in datasets:
    for datatype in datatypes:
        if not os.path.exists( './'+ data_path + "/" + dataset + "/" + datatype):
            os.makedirs( './'+ data_path + "/" + dataset + "/" + datatype)

# Get all cat and dog files based on regular expression. Ex.: cat.xxx
list_cat = glob.glob("./dataset/cat*")
list_dog = glob.glob("./dataset/dog*")

# Data distribution
# We chose 70/15/15 as our train/test/validation distribution which is standard practice.
images_per_class = 1500         # Images per dog and cat class
train_distribution = 0.7        # Train distribution
test_distribution = 0.15        # Test distribution
validation_distribution = 0.15  # Validation distribution

# Calculate our distributions as numbers
train_image_count = int(images_per_class * train_distribution)
test_image_count = int(images_per_class * test_distribution)
validation_image_count = int(images_per_class * validation_distribution)

# Takes k unique samples from a given list
train_dog_images = random.sample(list_dog, k=train_image_count)
train_cat_images = random.sample(list_cat, k=train_image_count)

# We convert our lists to set so we can substract previous chosen samples
test_dog_images = random.sample(list(set(list_dog)-set(train_dog_images)), k=test_image_count)
test_cat_images = random.sample(list(set(list_cat)-set(train_cat_images)), k=test_image_count)

# We subsract the previous two
validation_dog_images = random.sample(list(set(list_dog)-set(train_dog_images+test_dog_images)), k=validation_image_count)
validation_cat_images = random.sample(list(set(list_cat)-set(train_cat_images+test_cat_images)), k=validation_image_count)

# Checking the correct file count
print(f"Train : Dog len {len(train_dog_images)} and cat len {len(train_cat_images)}")
print(f"Test : Dog len {len(test_dog_images)} and cat len {len(test_cat_images)}")
print(f"Test : Dog len {len(validation_dog_images)} and cat len {len(validation_dog_images)}")

# Strip string so only filename is left
def strip_filename(filename):
    return filename.split("/")[-1].replace("\\", "")

# Distribute our files into different folders
# Do training data for dog
for file in train_dog_images:
    file = strip_filename(file)
    target_path = f"{data_path}/training/dogs/{file}"
    file_path = f"./dataset/{file}"
    shutil.copyfile(file_path, target_path)

# Do training data for cat
for file in train_cat_images:
    file = strip_filename(file)
    target_path = f"{data_path}/training/cats/{file}"
    file_path = f"./dataset/{file}"
    shutil.copyfile(file_path, target_path)

# Do test data for dog
for file in test_dog_images:
    file = strip_filename(file)
    target_path = f"{data_path}/test/dogs/{file}"
    file_path = f"./dataset/{file}"
    shutil.copyfile(file_path, target_path)

# Do test data for cat
for file in test_cat_images:
    file = strip_filename(file)
    target_path = f"{data_path}/test/cats/{file}"
    file_path = f"./dataset/{file}"
    shutil.copyfile(file_path, target_path)

# Do validation data for cat
for file in validation_dog_images:
    file = strip_filename(file)
    target_path = f"{data_path}/validation/dogs/{file}"
    file_path = f"./dataset/{file}"
    shutil.copyfile(file_path, target_path)

# Do validation data for dog
for file in validation_cat_images:
    file = strip_filename(file)
    target_path = f"{data_path}/validation/cats/{file}"
    file_path = f"./dataset/{file}"
    shutil.copyfile(file_path, target_path)

# Finito >:D
