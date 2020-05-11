from pysmiles import read_smiles

def oneHot(smiles):
	mol = read_smiles(smiles)
	mol_with_H = read_smiles(smiles, explicit_hydrogen=True)

	one_hot_matrix = []
	for atom in mol.nodes(data='element'):
		row = [0]*4
		if atom[1]=="C":
			row[0]=1
		elif atom[1]=="N":
			row[1]=1
		elif atom[1]=="O":
			row[2]=1
		one_hot_matrix.append(row)
	for x in range(len(mol), len(mol_with_H)):
		one_hot_matrix.append([0,0,0,1])

	# for i in one_hot_matrix:
		# print(i)
	return one_hot_matrix