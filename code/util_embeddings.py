import os
from os.path import join
from rdkit import Chem
import numpy as np

BASE_DIR = join(".")

aa = set("abcdefghiklmnpqrstxvwyzv".upper())



def create_empty_path(path):
	try:
		os.mkdir(path)
	except:
		pass

	all_files = os.listdir(path)
	for file in all_files:
		os.remove(join(path, file))


		aa = set("abcdefghiklmnpqrstxvwyzv".upper())

def validate_enzyme(seq, alphabet=aa):
    "Checks that a sequence only contains values from an alphabet"
    leftover = set(seq.upper()) - alphabet
    return not leftover


def convert_to_smiles(met):
    if is_KEGG_ID(met):
        try:
            met = Chem.MolToSmiles(Chem.MolFromMolFile(join(BASE_DIR, "data", "mol-files", met + ".mol")))
        except:
            met = np.nan
    elif is_InChI(met):
        try:
            met = Chem.MolToSmiles(Chem.MolFromInchi(met))
        except:
            met = np.nan
    elif is_SMILES(met):
        None
    else:
        met = np.nan
    return(met)


def is_KEGG_ID(met):
    #a valid KEGG ID starts with a "C" or "D" followed by a 5 digit number:
    if len(met) == 6 and met[0] in ["C", "D"]:
        try:
            int(met[1:])
            return(True)
        except: 
            pass
    return(False)

def is_SMILES(met):
    m = Chem.MolFromSmiles(met,sanitize=False)
    if m is None:
      return(False)
    else:
      try:
        Chem.SanitizeMol(m)
      except:
        print('.......Metabolite string "%s" is in SMILES format but has invalid chemistry' % met)
        return(False)
    return(True)

def is_InChI(met):
    m = Chem.inchi.MolFromInchi(met,sanitize=False)
    if m is None:
      return(False)
    else:
      try:
        Chem.SanitizeMol(m)
      except:
        print('......Metabolite string "%s" is in InChI format but has invalid chemistry' % met)
        return(False)
    return(True)


