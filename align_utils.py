from rdkit import Chem
from itertools import product
from utils import rearrange
from rdkit.Chem import rdmolops
def align_mol_to_frags(smi_molecule, smi_linker, smi_frags):
    try:
        # Load SMILES as molecules
        mol = Chem.MolFromSmiles(smi_molecule)
        frags = Chem.MolFromSmiles(smi_frags)
        linker = Chem.MolFromSmiles(smi_linker)
        # transform aromatic rings into single-double format
        # rdmolops.Kekulize(mol, clearAromaticFlags=True)
        # rdmolops.Kekulize(frags, clearAromaticFlags=True)
        # rdmolops.Kekulize(linker, clearAromaticFlags=True)
        # Include dummy atoms in query
        du = Chem.MolFromSmiles('*')
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True
    
        # Renumber molecule based on frags (incl. dummy atoms)
        aligned_mols = []

        sub_idx = []
        # Get matches to fragments and linker
        qfrag = Chem.AdjustQueryProperties(frags, qp)
        # rdmolops.Kekulize(qfrag, clearAromaticFlags=True)
        frags_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
        qlinker = Chem.AdjustQueryProperties(linker, qp)
        # rdmolops.Kekulize(qlinker, clearAromaticFlags=True)
        linker_matches = list(mol.GetSubstructMatches(qlinker, uniquify=False))

        # Loop over matches
        for frag_match, linker_match in product(frags_matches, linker_matches):
            # Check if match
            f_match = [idx for num, idx in enumerate(frag_match) if frags.GetAtomWithIdx(num).GetAtomicNum() != 0]
            l_match = [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in f_match]
            # If perfect match, break
            if len(set(list(f_match)+list(l_match))) == mol.GetNumHeavyAtoms():
                break
        # Add frag indices
        sub_idx += frag_match
        # Add linker indices to end
        sub_idx += [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]

        aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))
        aligned_mols.append(frags)
        re_idx = sub_idx.copy()

        nodes_to_keep = [i for i in range(len(frag_match))]
        
        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        for i, al_mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
            for idx in dummy_idx:
                sub_idx.remove(idx)
                sub_idx.append(idx)
            if i == 0:
                mol_range = list(range(al_mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(al_mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(al_mol, sub_idx)
            if i == 0:
                re_idx_final = rearrange(re_idx, sub_idx)

        # Get exit vectors
        exit_vectors = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())

        if len(exit_vectors) != 2:
            print("Incorrect number of exit vectors")

        return (aligned_mols[0], aligned_mols[1]), nodes_to_keep, exit_vectors, re_idx_final

    except:
        # print("Could not align")
        return ([],[]), [], [], []


