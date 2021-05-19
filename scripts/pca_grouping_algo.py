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
columns = []

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