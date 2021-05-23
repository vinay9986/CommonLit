import random
random.seed(83174236)

class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2

def generateDNAMotif(columns, n):
    if n > 26:
        print("n cannot be more than 26")
        return np.nan
    mapping_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    random.shuffle(columns)
    matrix_list = []
    for i in range(0, len(columns), n):
        sub_set= columns[i:i+n]
        col_key_map = TwoWayDict()
        matrix = {}
        for index, i in enumerate(sub_set):
            col_key_map[i] = mapping_keys[index]
        for col_a in sub_set:
            col_corr = []
            for col_b in [x for x in sub_set if x != col_a]:
                col_corr.append((col_b, abs(train_feat[col_a].corr(train_feat[col_b]))))
            matrix[col_a] = sorted(col_corr, key=lambda x: x[1], reverse=True)
        matrix_list.append((col_key_map, matrix))
    return matrix_list

def convertDNAToPCAGroup(column_key_map, dnas):
    pca_groups = []
    for dna in dnas:
        pca_group = []
        for i in dna:
            pca_group.append(column_key_map.get(i))
        pca_groups.append(pca_group)
    return pca_groups

def generateDNAString(columns, n=5):
    matrix_list = generateDNAMotif(columns, n)
    dnas_list = []
    for i in matrix_list:
        column_key_map = i[0]
        dnas = []
        for key in i[1]:
            dnas.append(column_key_map.get(key) + ''.join([column_key_map.get(value[0]) for value in i[1][key]]))
        dnas_list.append((column_key_map, dnas))
    return dnas_list, matrix_list

# cell 2
num_cols = list(train_feat.select_dtypes(exclude=['object']).columns)
num_cols.remove('target')
num_cols.remove('standard_error')
pos_corr = []
neg_corr = []
for col in num_cols:
    corr = train_feat[col].corr(train_feat['target'])
    if corr >= 0:
        pos_corr.append((col, round(corr, 4)))
    else:
        neg_corr.append((col, round(abs(corr), 4)))

boostable_pos_fe = [item[0] for item in pos_corr if .25 <= item[1] < .40]
boostable_neg_fe = [item[0] for item in neg_corr if .25 <= item[1] < .40]

print(len(boostable_pos_fe))
# output of the above code is 36
print(len(boostable_neg_fe))
# output of the above code is 43

# cell 3
# divisors of 36
pos_pca_groups = getPCAGroups(boostable_pos_fe, [4, 6, 9, 12, 18])
# divisors of 40, because 43 is prime number
neg_pca_groups = getPCAGroups(boostable_neg_fe, [4, 5, 8, 10, 20])
print("groups created for analysis!")

# cell 4
# PCA analysis