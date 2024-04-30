import pandas as pd
import matplotlib.pyplot as plt
from Functions import *
from config import *
from copy import deepcopy


# LONGCOVID3 status within HIV status 2, 1, or 3
new_set = pd.read_csv("new_set.csv", header=0, index_col=0)

#Reading Formatted Metabolomics Dataset
data = pd.read_csv("formatted_dataset.csv", header=0, index_col=0, encoding="ISO-8859-1")  #Subject ID is Index

# Replacing columns in data with columns from new_set
data.update(new_set[attribute_cols])

data_orig = deepcopy(data)


#This analysis is for EVERHOSP 1 vs 0

feature_groups_dict = {
		"METABOLITES-1": METABOLITES_1,
		"DRUGS-1": DRUGS_1,
		"RATIOS-1": RATIOS_1,
		"METABOLITES-2": METABOLITES_2,
		"DRUGS-2": DRUGS_2,
		"RATIOS-2": RATIOS_2,
		"INFLAMMATORY MARKERS": INFLAMMATORY_MARKERS
	}
feature_groups_list= [
		"METABOLITES-1",
		"DRUGS-1",
		"RATIOS-1",
		"METABOLITES-2",
		"DRUGS-2",
		"RATIOS-2",
		#"INFLAMMATORY MARKERS"
]

header =["PARAMETER", "FEATURE_GROUP", "NAME", "IF Detected my 1 & 2", "SCORE", "P-VALUE", "MEAN-0", "MEAN-1", "MEDIAN-0", "MEDIAN-1"]
mean_header = ["NAME", "MEAN-0", "MEAN-1", "MEDIAN-0", "MEDIAN-1", "FEATURE_GROUP", "SCORE", "PARAMETER", "Pvalue", "STDev-0", "STDev-1", "SME-0", "SME-1"]

all_the_features = []
all_the_features.extend(METABOLITES_2)
all_the_features.extend(DRUGS_2)
all_the_features.extend(RATIOS_2)
all_the_features.extend(INFLAMMATORY_MARKERS)
all_the_features = [name for name in all_the_features if name not in drop_columns]

data = data.drop(columns=drop_columns)
print(data.shape)

data = data[data["EVERPOS"] == 1]
print("EVERPOS =1 Filter:", data.shape)

######
#Condition 1
#data = data[data["HIV_STAT"] == 2]
#case_name = "HIV_STAT 2"
#Condition 2
#data = data[(data["HIV_STAT"] == 1) | (data["HIV_STAT"] == 3)]
#case_name = "HIV_STAT 1 & 3"
#Condition 3
data = data[(data["HIV_STAT"] == 1) | (data["HIV_STAT"] == 2) | (data["HIV_STAT"] == 3)]
case_name = "HIV_STAT 1 2 & 3"
#Condition 4
#data = data[(data["HIV_STAT"] == 2) | (data["HIV_STAT"] == 3)]
#case_name = "HIV_STAT 2 & 3"

met_to_rm = []

#Long Covid Analysis
data =  data[data["LONGCOVID3"].isin([1, 0])]
parameter = ['LONGCOVID3']
#
#Ever Hospitalization Analysis
#data =  data[data["EVERHOSP"].isin([1, 0])]
#parameter = ['EVERHOSP']

if case_name == "HIV_STAT 2 & 3" and parameter[0] == "EVERHOSP":
	met_to_rm = ["Methylimidazoleacetic acid (mzCloud ID 1)#2"]


#Red -> Upregulated
#Blue -> Downregulated

print("Original Data:", len(data.index.values.tolist()), data[parameter[0]].value_counts().to_dict())

# Sampling The Data for ML 
# Over => Oversampling Under => Undersampling  All => AllData (Unbalanced)
sampling = "Over"		#Over|Under|All


csids_to_test = []

if "EVERHOSP" in parameter:
	csids_to_test = csids_to_test_everhosp
elif "LONGCOVID3" in parameter:
	csids_to_test = csids_to_test_longcovid3

print(f"Sampling Type: {sampling}")


columns_to_keep = []
for fgroup in feature_groups_list:
	colnames = feature_groups_dict[fgroup][:]
	columns_to_keep.extend(colnames)
columns_to_keep.extend(parameter)


for param in parameter:

	all_the_features.append(param)
	FGROUP_FEATURE_DICT = {}

	data1 = deepcopy(data)
	data1 = data1[data1[param].notnull()]

	data1 = data1[columns_to_keep]

	data1, null_sum = impute_outliers(data1, attribute_cols)

	test_data = data1.copy()

	df_oversampled  = oversampling(deepcopy(data1), param)
	
	print("Oversampled Data:", len(df_oversampled.index.values.tolist()), df_oversampled[param].value_counts().to_dict())

	mean_list_combined = []
	all_feature_dict = {}

	for fgroup in feature_groups_list:
		title = f"{param}-{fgroup}-log"

		feature_dict = {}
		colnames = feature_groups_dict[fgroup][:]
		
		if sampling == "All":
			feature_dict, df_wbest_feature = analyze_features(deepcopy(data1), colnames, param)
		elif sampling == "Over":
			feature_dict, df_wbest_feature = analyze_features(deepcopy(data1), colnames, param, met_to_rm)
			print("Features from Oversampling:", feature_dict)
		elif sampling == "Under":
			feature_dict, df_wbest_feature  = analyze_features_equalset(deepcopy(df_oversampled), colnames, param, sampling)

		for kyy in feature_dict:
			if kyy not in all_feature_dict:
				all_feature_dict[kyy] = []
			all_feature_dict[kyy] = feature_dict[kyy]
			all_feature_dict[kyy].append(fgroup)

		title = f"{case_name}-{param}-{sampling}-log"

		mean_med_dict, mean_df = calculateMeanMedian(deepcopy(df_wbest_feature), param, feature_dict, fgroup)

		for key in mean_med_dict:
			mean_list_combined.append(mean_med_dict[key])

	mean_df =pd.DataFrame(mean_list_combined, columns=mean_header)
	mean_df["MeanDiff"] = mean_df["MEAN-1"] - mean_df["MEAN-0"]
	mean_df.to_csv(f"images_out/MeanDiff-{case_name}-{param}.csv", index=False)
	###Machine Learning Data
	if sampling == "Over":		#Oversampled Data
		selected_features = list(all_feature_dict.keys())
		selected_features.append(param)
		data_for_boxplot = deepcopy(data1)
		data_for_boxplot_log = df_log_transform(data_for_boxplot, param)
		data_for_boxplot_log = data_for_boxplot_log[selected_features]
		#drawboxplot(data_for_boxplot_log, title, param)
		
		print("selected_features K-Best: ", selected_features, len(selected_features))
		df_wbest_feature = df_oversampled[selected_features]
		data_for_ml = deepcopy(df_wbest_feature)
		
		#Log Transformation of The dataset
		data_for_ml = df_log_transform(data_for_ml, param)
		#Scaling the dataset
		data_for_ml_log_equal = scale_data(data_for_ml, param)
		data_for_ml_log_equal, feat_list = doRFE(data_for_ml_log_equal, param)
  
		feat_list = data_for_ml_log_equal.columns.tolist()
  
		plot_tSNE(data_for_ml_log_equal, title)
  
		vertical_scatter_plot_errorbar(deepcopy(mean_df), title, title, feat_list)
  
		print(case_name, parameter)
		print("Final Feature used for ML: ", feat_list)

		accuracy, roc_auc, prc_auc, fpr, tpr, recall, precision, model_filename, sensitivity, specificity, precisionsc, f1 = ML_model(data_for_ml_log_equal, title, param)

		print("Testing with Real Data")
		testML_realdata(model_filename, feat_list, param, test_data, case_name)

	print("Analysis Finished!")
	"""
	#The following analysis is not required anymore

	pca_df = plotPCA(data_for_ml_log_equal, title, param)
	print(pca_df.head())
	clustering(pca_df, title, param)
	print("selected_features", len(data_for_ml_log_equal.columns.tolist()))
	plotRFE(data_for_ml_log_equal)
	featuretool_processing(data_for_ml_log_equal, param)

	met_common, ratio_common, drugs_common = get_common_metabolite(FGROUP_FEATURE_DICT)
	met_common = ";".join(met_common)
	print(met_common, ratio_common, drugs_common)

	try:
		draw_violinplot(df, feature_dict, param, "{}-{}-log".format(fgroup, param), "{}-{}-log".format(fgroup, param))
	except Exception as Err:
		print(Err)
	"""


exit()

"""
#Following Analysis is Not Required anymore
#Log Transform
###ANALYZING INDIVIDUAL FEATURE-SETS######
METABOLITES_1_Top10 = [METABOLITES_1[i] for i in (168,152,58,142,200,185,178,143,56,136)]
METABOLITES_1_Bottom10  = [METABOLITES_1[i] for i in (120,37,208,153,203,151,0,88,59,71)]

METABOLITES_2_Top10 = [METABOLITES_2[i] for i in (152,168,142,58,200,148,86,143,139,154)]
METABOLITES_2_Bottom10 = [METABOLITES_2[i] for i in (169,43,206,204,145,180,83,234,150,88)]

#draw_violinplot(data, METABOLITES_2_Top10, "METABOLITES_2_Top10", attribute_cols)
#draw_violinplot(data, METABOLITES_2_Bottom10, "METABOLITES_2_Bottom10", attribute_cols)

feature_dict = analyze_features(data, METABOLITES_1, attribute_cols, "METABOLITES_1_Top")

#print("METABOLITES_1_Top", feature_dict, len(feature_dict))

lst = METABOLITES_1
lst.append("EVERHOSP")
datatmp = data[lst]
datatmp = datatmp.astype({"EVERHOSP": int})
scores = random_forest_classification(datatmp)
print("Accuracy", round(scores.mean(),2), round(scores.std(),2))
##RESULT: Accuracy 0.87 0.01

for i in range(30):
	newdata = Calc_Feature_Importance(data, feature_dict, attribute_cols, i+1)
	scores, tn, fp, fn, tp= random_forest_classification(newdata)
	print(i, tn, fp, fn, tp)
	#print("Accuracy:{}".format(i+1), round(scores.mean(),2), round(scores.std(),2))
#newdata =  kendall_corr(newdata)
exit()
#newdata = doRFE(data)

plot_tSNE(newdata)
exit()
input("")
feature_dict = analyze_features(data, METABOLITES_2, attribute_cols, "METABOLITES_2_Top")
print("METABOLITES_2_Top", feature_dict, len(feature_dict))
input("")
feature_dict = analyze_features(data, DRUGS_1, attribute_cols, "DRUGS_1_Top")
print("DRUGS_1_Top", feature_dict, len(feature_dict))
input("")
feature_dict = analyze_features(data, DRUGS_2, attribute_cols, "DRUGS_2_Top")
print("DRUGS_2_Top", feature_dict, len(feature_dict))
input("")
feature_dict = analyze_features(data, INFLAMMATORY_MARKERS, attribute_cols, "INFLAMMATORY_MARKERS_Top")
print("INFLAMMATORY_MARKERS_Top", feature_dict, len(feature_dict))
input("")
exit()
feature_bar_plot(feature_dict, "METABOLITES_1_Top", "Metabolites-1 (P-value <=0.05)")
draw_violinplot(data, feature_dict, attribute_cols, "METABOLITES_1_Top", "Metabolites-1 (P-value <=0.05)")


feature_bar_plot(feature_dict, "METABOLITES_2_Top", "Metabolites-2 (P-value <=0.05)")
draw_violinplot(data, feature_dict, attribute_cols, "METABOLITES_2_Top", "Metabolites-2 (P-value <=0.05)")



feature_bar_plot(feature_dict, "DRUGS_1_Top", "DRUGS-1 (P-value <=0.05)")
draw_violinplot(data, feature_dict, attribute_cols, "DRUGS_1_Top", "DRUGS-1 (P-value <=0.05)")



feature_bar_plot(feature_dict, "DRUGS_2_Top", "DRUGS-2 (P-value <=0.05)")
draw_violinplot(data, feature_dict, attribute_cols, "DRUGS_2_Top", "DRUGS-2 (P-value <=0.05)")

feature_dict = analyze_features(data, RATIOS_1, attribute_cols, "RATIOS_1_Top")
if len(feature_dict) != 0:
	feature_bar_plot(feature_dict, "RATIOS_1_Top", "RATIOS-1 (P-value <=0.05)")
	draw_violinplot(data, feature_dict, attribute_cols, "RATIOS_1_Top", "RATIOS-1 (P-value <=0.05)")

feature_dict = analyze_features(data, RATIOS_2, attribute_cols, "RATIOS_2_Top")
if len(feature_dict) != 0:
	feature_bar_plot(feature_dict, "RATIOS_2_Top", "RATIOS-2 (P-value <=0.05)")
	draw_violinplot(data, feature_dict, attribute_cols, "RATIOS_2_Top", "RATIOS-2 (P-value <=0.05)")

feature_dict = analyze_features(data, INFLAMMATORY_MARKERS, attribute_cols, "INFLAMMATORY_MARKERS_Top")
feature_bar_plot(feature_dict, "INFLAMMATORY_MARKERS_Top", "INFLAMMATORY_MARKERS (P-value <=0.05)")
draw_violinplot(data, feature_dict, attribute_cols, "INFLAMMATORY_MARKERS_Top", "INFLAMMATORY_MARKERS (P-value <=0.05)")
"""