import gzip
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# creates headers
X_headers = ['Morgan_' + str(idx) for idx in range(128)]
X_headers.append("smiles")

with open('train-morgan-128.csv', 'w') as train_out:
    train_out_csv = csv.writer(train_out, delimiter=',')
    train_out_csv.writerow(X_headers)
    
    with gzip.open('../data/train.gz', 'r') as train_fh:
        train_csv = csv.reader(train_fh, delimiter=',')
        next(train_csv)
        count = 0

        for row in train_csv: 
            # prints count to track progress
            if (count % 10000 == 0):
                print count
            count += 1

            # save the smile
            smile = row[0]

            # processes smile into molecule, produces fingerprints
            mol = AllChem.MolFromSmiles(smile)

            features = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 128)
            features = np.asarray(features, dtype = int).tolist()
            features.append(smile)

            train_out_csv.writerow(features)
