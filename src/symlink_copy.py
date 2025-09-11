# AI-generated Code

import os

source_dir = "/media/datasets/dataset-myelom/atlas/atlas" 
data_dir = "/home/student/alissa/3dLegPois/src/dataset/data_preprocessing/atlas-dataset-folder" 

rawdata_dir = os.path.join(data_dir, "rawdata")
derivatives_dir = os.path.join(data_dir, "derivatives")
os.makedirs(rawdata_dir, exist_ok=True)
os.makedirs(derivatives_dir, exist_ok=True)


for filename in os.listdir(source_dir):
    # Extract subject and session from filename: 'sub-<subject>_ses-<session>_<rest_of_filename>'
    parts = filename.split('_')

    if len(parts) < 2:
        print(f"Warning: File {filename} isn't in the expected format. Skip.")
        continue
    
    subject = parts[0]  # sub-CTFU00099
    session = parts[1]  # ses-20191209
    
    # Create directories (subject, session) in rawdata and derivatives
    rawdata_subject_dir = os.path.join(rawdata_dir, subject, session)
    derivatives_subject_dir = os.path.join(derivatives_dir, subject, session)

    os.makedirs(rawdata_subject_dir, exist_ok=True)
    os.makedirs(derivatives_subject_dir, exist_ok=True)

    if "_ct" in filename:
        if "_poi" not in filename:
            target_dir = rawdata_subject_dir
        else:
            target_dir = derivatives_subject_dir
    else:
        target_dir = derivatives_subject_dir
    
    # Create symlink
    source_file = os.path.join(source_dir, filename)
    symlink_path = os.path.join(target_dir, filename)

    if not os.path.exists(symlink_path):
        os.symlink(source_file, symlink_path)
    else:
        print(f"Symlink already exists: {symlink_path}")