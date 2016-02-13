import gzip
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# creates headers
X_headers = ['Morgan_' + str(idx) for idx in range(128)]
X_headers.append("ID")
X_headers.append("smiles")

with open('test-morgan-128.csv', 'w') as test_out:
    test_out_csv = csv.writer(test_out, delimiter=',')
    test_out_csv.writerow(X_headers)
    
    with gzip.open('../data/test.gz', 'r') as test_fh:
        test_csv = csv.reader(test_fh, delimiter=',')
        next(test_csv)
        count = 0

        for row in test_csv: 
            # prints count to track progress
            if (count % 10000 == 0):
                print count
            count += 1

            # save the smile
            smile = row[1]
	    ID = row[0]
            
	    # processes smile into molecule, produces fingerprints
            mol = AllChem.MolFromSmiles(smile)

            features = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 128)
            features = np.asarray(features, dtype = int).tolist()
	    features.append(ID)
            features.append(smile)

            test_out_csv.writerow(features)
