import os

# Setze die Pfade für dein Projekt
source_dir = "/media/datasets/dataset-myelom/atlas/atlas"  # Der Ordner mit den aktuellen Dateien
data_dir = "/home/student/alissa/3dLegPois/src/dataset/data_preprocessing/atlas-dataset-folder"  # Der Zielordner, in dem die Symlinks erstellt werden

# Erstelle die Hauptordner, falls sie nicht existieren
rawdata_dir = os.path.join(data_dir, "rawdata")
derivatives_dir = os.path.join(data_dir, "derivatives")
os.makedirs(rawdata_dir, exist_ok=True)
os.makedirs(derivatives_dir, exist_ok=True)


# Durchlaufe alle Dateien im Quellordner
for filename in os.listdir(source_dir):
    # Extrahiere Subjekt und Session aus dem Dateinamen
    # Annahme: Dateiname folgt dem Muster 'sub-<subject>_ses-<session>_<rest_of_filename>'
    parts = filename.split('_')

    if len(parts) < 2:
        print(f"Warnung: Datei {filename} entspricht nicht dem erwarteten Format. Wird übersprungen.")
        continue
    
    subject = parts[0]  # sub-CTFU00099
    session = parts[1]  # ses-20191209
    
    # Erstelle die Verzeichnisse für das Subjekt und die Session unter rawdata und derivatives
    rawdata_subject_dir = os.path.join(rawdata_dir, subject, session)
    derivatives_subject_dir = os.path.join(derivatives_dir, subject, session)

    os.makedirs(rawdata_subject_dir, exist_ok=True)
    os.makedirs(derivatives_subject_dir, exist_ok=True)

    # Bestimme den Dateityp anhand des Namens (CT, POI, Masken)
    if "_ct" in filename:
        # Falls es sich um eine CT-Datei handelt (keine POI-Datei)
        if "_poi" not in filename:
            target_dir = rawdata_subject_dir
        else:
            # Wenn es sich um eine POI-Datei handelt, auch wenn '_ct' im Namen enthalten ist
            target_dir = derivatives_subject_dir
    else:
        # Alle anderen Dateien, die keine '_ct' im Namen haben, gehören zu derivatives
        target_dir = derivatives_subject_dir
    
    # Erstelle den Symlink für die Datei im entsprechenden Zielordner
    source_file = os.path.join(source_dir, filename)
    symlink_path = os.path.join(target_dir, filename)

    # Falls der Symlink noch nicht existiert, erstelle ihn
    if not os.path.exists(symlink_path):
        os.symlink(source_file, symlink_path)
        #print(f"Symlink erstellt: {symlink_path}")
    else:
        print(f"Symlink existiert bereits: {symlink_path}")