# cell 1
# input for step 1 and 2
all_independent_columns = train_feat.columns.to_list()
all_independent_columns.remove('target')
all_independent_columns.remove('standard_error')
all_independent_columns.remove('excerpt')

for index, i in enumerate(range(0, len(all_independent_columns), 79)):
    train_cols = all_independent_columns[i:i+79]
    train_cols.append('target')
    test_cols = all_independent_columns[i:i+79]
    report = sv.compare(train_feat[train_cols], test_feat[test_cols], 'target')
    file_name = "EDA_report_"+str(index)+".html"
    report.show_html(file_name)

#cell 2
# input for step 3 and step 4 
# usually the iterations are for 10 features at a time and for second round if the features are close to 10 (10+3) then modify the iteration so that only one report is generated
columns = []

for index, i in enumerate(range(0, len(columns), 10)):
    train_cols = columns[i:i+10]
    train_cols.append('target')
    test_cols = columns[i:i+10]
    report = sv.compare(train_feat[train_cols], test_feat[test_cols], 'target')
    file_name = "R2_EDA_GT30_problem_feat_"+str(index)+".html"
    report.show_html(file_name)

#cell 3
# input for step 5
random.seed(3456789)
columns = []
random.shuffle(columns)

for index, i in enumerate(range(0, len(columns), 10)):
    train_cols = columns[i:i+10]
    train_cols.append('target')
    test_cols = columns[i:i+10]
    report = sv.compare(train_feat[train_cols], test_feat[test_cols], 'target')
    file_name = "EDA_GT30_grouping_analysis_"+str(index)+".html"
    report.show_html(file_name)

#cell 4
# step 6 helper
deep_copy_train = copy.deepcopy(train_feat)
deep_copy_test = copy.deepcopy(test_feat)

#cell 5
# step 6
pca1 = ['IN', 'nominalization']
pca_res1, train_feat = train_pca(train_feat, pca1, 'pca1')
test_feat = apply_pca(pca_res1, test_feat)

report = sv.compare(train_feat[['pca1_1', 'pca1_2', 'target']], test_feat[['pca1_1', 'pca1_2']], 'target')
report.show_html('pcaReport.html')

#cell 6
# step 6 helper (always run this step before generating new pcaReport.html)
train_feat = copy.deepcopy(deep_copy_train)
test_feat = copy.deepcopy(deep_copy_test)

###############################################################################################
##                                    IMPROVED STEP 6                                        ##
###############################################################################################

# cell 5
#improved step 6
deep_copy_train = copy.deepcopy(train_feat)
deep_copy_test = copy.deepcopy(test_feat)

pca_analysis_map = {}
pca_analysis_map['pca1'] = ['spacy_24', 'spacy_10']
pca_analysis_map['pca2'] = ['gunning_fog', 'lix', 'words_per_sent']
pca_analysis_map['pca3'] = ['gunning_fog', 'lix', 'words_per_sent', 'spacy_24']
pca_analysis_map['pca4'] = ['gunning_fog', 'lix', 'words_per_sent', 'spacy_24', 'spacy_10']
pca_analysis_map['pca5'] = ['gunning_fog', 'lix', 'words_per_sent', 'spacy_24', 'spacy_10', 'kincaid']
pca_analysis_map['pca6'] = ['spacy_9', 'spacy_20']
pca_analysis_map['pca7'] = ['spacy_133', 'spacy_184']
pca_analysis_map['pca8'] = ['spacy_139', 'spacy_147', 'spacy_133']
pca_analysis_map['pca9'] = ['spacy_139', 'spacy_147', 'spacy_133', 'spacy_67']
pca_analysis_map['pca10'] = ['spacy_139', 'spacy_147', 'spacy_133', 'spacy_67', 'spacy_156']
pca_analysis_map['pca11'] = ['spacy_84', 'spacy_162']
pca_analysis_map['pca12'] = ['spacy_84', 'spacy_162', 'spacy_155']
pca_analysis_map['pca13'] = ['spacy_208', 'spacy_232']
pca_analysis_map['pca14'] = ['spacy_208', 'spacy_232', 'spacy_262']
pca_analysis_map['pca15'] = ['spacy_208', 'spacy_232', 'spacy_262', 'spacy_261']
pca_analysis_map['pca16'] = ['spacy_200', 'spacy_263']
pca_analysis_map['pca17'] = ['spacy_298', 'VB']
pca_analysis_map['pca18'] = ['spacy_298', 'VB', 'WRB']
pca_analysis_map['pca19'] = ['WDT', 'JJ']
pca_analysis_map['pca20'] = ['spacy_4', 'pronoun']
pca_analysis_map['pca21'] = ['exclaims', 'questions', 'pronoun']
pca_analysis_map['pca22'] = ['spacy_7', 'spacy_27']
pca_analysis_map['pca23'] = ['spacy_103', 'spacy_97']
pca_analysis_map['pca24'] = ['spacy_103', 'spacy_97', 'spacy_159']
pca_analysis_map['pca25'] = ['spacy_103', 'spacy_97', 'spacy_159', 'spacy_60']
pca_analysis_map['pca26'] = ['spacy_86', 'spacy_105']
pca_analysis_map['pca27'] = ['spacy_160', 'spacy_203']
pca_analysis_map['pca28'] = ['spacy_294', 'spacy_196']

for key in pca_analysis_map:
    pca_res, train_feat = train_pca(train_feat, pca_analysis_map[key], key)
    test_feat = apply_pca(pca_res, test_feat)
    report = sv.compare(train_feat[[key+'_1', key+'_2', 'target']], test_feat[[key+'_1', key+'_2']], 'target')
    file_name = "EDA_motif_pca_grouping_R1_"+key+".html"
    report.show_html(file_name)
    train_feat = copy.deepcopy(deep_copy_train)
    test_feat = copy.deepcopy(deep_copy_test)

