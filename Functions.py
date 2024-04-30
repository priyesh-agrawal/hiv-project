import pandas as pd
import numpy as np
import re
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_selection import f_classif, SelectKBest, RFE, mutual_info_classif, mutual_info_regression
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (cross_val_score, StratifiedKFold, train_test_split, cross_val_predict, 
                                    GridSearchCV, RepeatedStratifiedKFold)
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.datasets import make_classification
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, accuracy_score, 
                            precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, 
                            auc)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.utils import resample
import random
from imblearn.over_sampling import SMOTE 
from copy import deepcopy
import warnings
from woodwork.logical_types import Categorical
import featuretools as ft
#python -m venv env

color_dict = {
		"METABOLITES-2":"Blue",
		"DRUGS-2": "Green",
		"RATIOS-2": "Purple",
		"INFLAMMATORY MARKERS": "Red"
}
serial_dict = {
		"METABOLITES-2":2,
		"DRUGS-2": 3,
		"RATIOS-2": 1,
		"INFLAMMATORY MARKERS": 4
}


def clustering(data, figname, param):
	# Extract the features from the DataFrame
	X = data.iloc[:, :-1].values

	# Fit spectral clustering to the data
	spectral = SpectralClustering(n_clusters=10, affinity='rbf', n_init=10, gamma=1)
	labels = spectral.fit_predict(X)

	# Add the cluster labels to the original DataFrame
	data['cluster'] = labels
	colors = ['red', 'blue']
	targets = data[param].values.tolist()
	colors_new = [colors[int(i)] for i in targets]
	
	#Plot the results
	fig, ax = plt.subplots()
	scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=colors_new, cmap='viridis')
	legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
	ax.add_artist(legend)

	plt.title('Spectral Clustering Results')

	plt.savefig("images_out/clustering-{}.png".format(figname))


def scale_data(df, target_colname):
	#return df
	#print(df.head())
	#input("aaac")
	target_col = df[target_colname]
	feature_cols = df.drop(target_colname, axis=1)
	# Create a scaler object and fit it to the dataframe
	scaler = StandardScaler()	#MinMaxScaler()
	scaler.fit(feature_cols)

	# Transform the dataframe using the scaler
	feature_cols_scaled = pd.DataFrame(scaler.transform(feature_cols), columns=feature_cols.columns, index=feature_cols.index)

	# Concatenate the target column with the scaled feature columns
	df_scaled = pd.concat([feature_cols_scaled, target_col], axis=1)

	return df_scaled

def draw_ridgeplot(df):
	#ax = sns.histplot(data=df, x="K/T#1", hue="EVERHOSP", element="step", stat="density", multiple="stack")
	
	sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
	g = sns.FacetGrid(df, row="EVERHOSP", hue="EVERHOSP", aspect=15, height=0.5, palette="Set2")
	g.map(sns.kdeplot, "K/T#1", clip_on=False, fill=True, alpha=1, lw=1.5, bw_method=.2)
	g.map(sns.rugplot, "K/T#1", height=-.05, clip_on=False)
	g.fig.subplots_adjust(hspace=-0.3)
	
	plt.savefig("images_out/my_plot.png")


def combined_bar_plot(list1, list2, list3 , size):


	categories = [f"{i}" for i in range(size)]
	values1 = list1
	values2 = list2
	values3 = list3
	# Creating the bar plot
	plt.plot(categories, values1, label='Accuracy', marker='o')
	plt.plot(categories, values2, label='PRC-AUC', marker='s')  # alpha is used to set the transparency of bars
	plt.plot(categories, values3, label='ROC-AUC', marker='^') 
	# Adding labels and title
	plt.xlabel('Iterations')
	plt.ylabel('Values')
	#plt.title('Bar Plot with 3 Lists')
	plt.legend()

	# Display the plot
	plt.savefig("images_out/accuracy_auc_plot.png")


def draw_violinplot(dfinp, feature_dict, parameter, figname, title):
	colnames = []
	colnames1 = []
	colnames = list(feature_dict.keys())
	colnames1 = list(feature_dict.keys())
	print(colnames1)
	colnames.append(parameter)
	print(dfinp.columns)
	max_value = dfinp[colnames1].max().max()
	min_value = dfinp[colnames1].min().min()

	collen = len(colnames1)-1
	sns.set(font_scale=2)

	fig, axs = plt.subplots(2, 5, figsize=(10, 8))
	#fig, axs = plt.subplots(2, 5, figsize=(10, 8))
	plt.subplots_adjust(wspace=1, hspace=0.4)
	plt.xticks(rotation=90)

	# Plot each group's boxplot on separate axes
	axs = axs.flatten()
	#print(axs)
	for i, (group_name, group_values) in enumerate(dfinp.items()):
		sns.boxplot(data=group_values, ax=axs[i])
		axs[i].set_title(group_name)
	plt.savefig("images_out/{}-boxplot-istart{}-2.png".format(figname, "equal"))
	
	"""
	for i,ax in enumerate(axs):
		print("i", i)
		plt.gca().set_xlabel('')
		sns.boxplot(data=dfinp, x=parameter, y=colnames1[i], ax=ax, palette="Set2")
			

		ylabel = ax.get_ylabel()
		ylabel = re.sub("#2", "",ylabel)
		wrapped_ylabel = '\n'.join(ylabel[i:i+30] for i in range(0, len(ylabel), 30))
				
		ax.set_xlabel('')
		ax.tick_params(labelsize=12)
		ax.yaxis.get_offset_text().set_fontsize(10)
		if colnames1[i] =="2-(3,4-dihydroxyphenyl)-3,5,7-trihydroxy-6-methyl-4H-chromen-4-one#1":
			ax.set_ylabel(wrapped_ylabel, fontsize=8)
		else:
			ax.set_ylabel(wrapped_ylabel, fontsize=10)
		ax.set_title("P-val {}".format(feature_dict[colnames1[i]][1]), fontsize=10)
		ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
	
		palette = sns.color_palette("Set2")
		categories = dfinp[parameter].unique()

		#handles = [plt.Rectangle((0,0),1,1, color=palette[i]) for i in range(len(categories))]
		#labels = categories

	#else:
	#	plt.gca().set_xlabel('')
		#print(df.head())
	#	sns.boxplot(data=dfinp, x=parameter, y=colnames1[0], ax=axs, palette="Set2")
	#	axs.set_xticklabels(axs.get_xticklabels(), rotation=0)
	#	axs.set_title("P-val {}".format(feature_dict[colnames1[0]][1]), fontsize=10)
	fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

	#plt.savefig("images/{}-boxplot-istart{}-new.png".format(figname, istart))
	plt.savefig("images_out/{}-boxplot-istart{}-2.png".format(figname, "equal"))
	plt.close()
	"""

def load_dataset(df):
	dataset = df.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y, df.columns.tolist()

def gen_random_color():
	import random
	# Generate a random RGB color
	red = random.randint(0, 255)
	green = random.randint(0, 255)
	blue = random.randint(0, 255)

	# Format the RGB color as a hexadecimal string
	color = '#{:02x}{:02x}{:02x}'.format(red, green, blue)
	return color

def feature_bar_plot_combined(feature_dict, figname, title):
	ax1 = plt.subplot()
	featurenames = []
	fscore = []
	color = []
	for key in feature_dict:
		key1  = key
		#key1 = re.sub("#\d+$", "", key1)
		key1 = re.sub("#\d+$|\s*\(mzCloud ID 1\)|\s*\(mzCloud ID 2\)", "", key1)
		featurenames.append(key1)
		fscore.append(feature_dict[key][0])
		color.append(color_dict[feature_dict[key][2]])

		plt.bar(featurenames, fscore, color=color)
	plt.ylabel("Feature Score")
	plt.xlabel("Feature Name")
	plt.title(title)
	plt.xticks(rotation=90)
	plt.savefig("images_out/{}_feature_bar.png".format(figname), bbox_inches="tight")
	plt.close()


def feature_bar_plot(feature_dict, figname, title):
	ax1 = plt.subplot()
	
	
	for i in range(1):
		score = []
		featurenames = []
		for key in feature_dict:
			key1  = key
			#key1 = re.sub("#\d+$", "", key1)
			key1 = re.sub("#\d+$|\s*\(mzCloud ID 1\)|\s*\(mzCloud ID 2\)", "", key1)
			featurenames.append(key1)
			score.append(feature_dict[key][i])
		#print(featurenames, score)
		#input("")
		plt.bar(featurenames, score, color="gray")
	plt.ylabel("Feature Score")
	plt.xlabel("Feature Name")
	plt.title(title)
	plt.xticks(rotation=90)
	plt.savefig("images_out/{}_feature_bar.png".format(figname), bbox_inches="tight")
	plt.close()


def add_constant_ifZero(dfinp, param):
	for col in dfinp.columns.tolist():
		if col != param:
			dfinp.loc[dfinp[col] == 0, col] += 0.001
	return dfinp

def df_log_transform(df, param):
	#df['Reg3a (pg/ml)'] = df['Reg3a (pg/ml)'].astype(float)

	df = add_constant_ifZero(deepcopy(df), param)

	
	df_log = df.loc[:, df.columns != param].apply(lambda x: np.log2(x))
	df_log[param] = df[param]

	return df_log

def analyze_features(df, colnames, parameter, met_to_rm):

	colnames_orig = colnames[:]
	if parameter not in colnames:
		colnames.append(parameter)

	df = df[colnames]
	df = df.astype(float)


	df = df_log_transform(df, parameter)

	#Finding Top most relevant features
	feature_dict, final_col = perform_kbest(df, parameter, colnames_orig)
	feature_dict = {key: value for key, value in feature_dict.items() if key not in met_to_rm}

	
	if len(final_col) == 1 and parameter in final_col:
		final_col = []
	final_col = [x for x in final_col if x not in met_to_rm]

	return feature_dict, df[final_col]


def analyze_features_equalset(df, colnames, parameter, sampling):

	colnames_orig = colnames[:]
	if parameter not in colnames:
		colnames.append(parameter)

	df = df[colnames]
	df = df.astype(float)

	df = df_log_transform(df, parameter)
	dict_of_features = {}
	if sampling == "Under":				#Undersampling

		for i in range(500):	
			#print(f"Iter: {i}")
			int_num =  random.randint(1, 100)
			df_loc = downsample_data(deepcopy(df), parameter, int_num)
			feature_dict, final_col = perform_kbest(df_loc, parameter, colnames_orig)
			for feat in final_col:
				if feat == parameter:
					continue
				if feat in dict_of_features:
					dict_of_features[feat] += 1
				else:
					dict_of_features[feat] = 0

		sorted_dict_descending = dict(sorted(dict_of_features.items(), key=lambda item: item[1], reverse=True))

		top_10_items = []
		for ky in sorted_dict_descending:
			if sorted_dict_descending[ky] >= 125:
				top_10_items.append(ky)

		final_col = top_10_items
		#input("")
		if len(final_col) == 1 and parameter in final_col:
			final_col = []
		feature_dict = {}
		for col_elm in final_col:
			feature_dict[col_elm] = [dict_of_features[col_elm], 0.05]
		final_col.append(parameter)
		return feature_dict, df[final_col]
	
	elif sampling == "Over":		#Oversampling
		df_loc = deepcopy(df)
		feature_dict, final_col = perform_kbest(df_loc, parameter, colnames_orig)
		#print("feature_dict", feature_dict, len(feature_dict))

		if len(final_col) == 1 and parameter in final_col:
			final_col = []
		return feature_dict, df_loc[final_col]		#Returning the oversampled DF instead of original one

def perform_kbest(df, parameter, colnames_orig):
	
	feature_dict = {}
	df = df.astype({parameter: int})

	X, y, featurenames = load_dataset(df)
	del featurenames[-1]

	fs = select_features(X, y)
	iterator = len(fs.scores_)
	#counter = 0
	for i in range(iterator):

		if fs.pvalues_[i] < 0.05:
			feature_dict[colnames_orig[i]] = [round(fs.scores_[i],4), round(fs.pvalues_[i], 4)]
	
	feature_dict = dict(sorted(feature_dict.items(), key=lambda item: item[1][0], reverse=True))
	final_col = list(feature_dict.keys())
	#final_col = final_col[0:5]
	keys = final_col
	final_col.append(parameter)
	feature_dict_new = {}
	for key in keys:
		if key in feature_dict:
			feature_dict_new[key] = feature_dict[key]
	#return small_dictionary
	return feature_dict_new, final_col

def select_features(X_train, y_train):

	fs = SelectKBest(score_func=f_classif, k="all")
	#fs = SelectKBest(score_func=mutual_info_regression, k='all')

	fs.fit(X_train, y_train)
	return fs

def calculateMeanMedian(data, param, feature_dict, fgroup):

	mean_dict = {}
	if param != "ALCGRP":
		mean_header = ["NAME", "MEAN-0", "MEAN-1", "MEDIAN-0", "MEDIAN-1", "FEATURE_GROUP", "SCORE", "PARAMETER", "Pvalue", "STDev-0", "STDev-1", "SEM-0", "SEM-1"]
	else:
		mean_header = ["NAME", "MEAN-0", "MEAN-1", "MEAN-2", "MEAN-3", "MEDIAN-0", "MEDIAN-1", "MEDIAN-2", "MEDIAN-3", "FEATURE_GROUP", "SCORE", "PARAMETER", "Pvalue"]
	mean_list = []
	for feature in feature_dict:
		feature_new = feature
		feature_new = re.sub("#\d+$|\s*\(mzCloud ID 1\)|\s*\(mzCloud ID 2\)", "", feature_new)
		#feature_new = re.sub("#\d+$", "", feature_new)
		mean = data.groupby(param)[feature].mean().to_dict()
		median = data.groupby(param)[feature].median().to_dict()
		stdev = data.groupby(param)[feature].std().to_dict()
		SEM = data.groupby(param)[feature].sem().to_dict()
		#input()
		mean_0 = mean[0]
		mean_1 = mean[1]
		median_0 = median[0]
		median_1 = median[1]
		stdev_0 = stdev[0]
		stdev_1 = stdev[1]
		SEM_0 = SEM[0]
		SEM_1 = SEM[1]
		if param == "ALCGRP":

			mean_2 = mean[2]
			mean_3 = mean[3]
			median_2 = median[2]
			median_3 = median[3]
			mean_dict[feature] = [feature_new, mean_0, mean_1, mean_2, mean_3, median_0, median_1, median_2, median_3, fgroup, feature_dict[feature][0], param]
			mean_list.append([feature_new, mean_0, mean_1, mean_2, mean_3, median_0, median_1, median_2, median_3, fgroup, feature_dict[feature][0], param])
		else:
			mean_dict[feature] = [feature_new, mean_0, mean_1, median_0, median_1, fgroup, feature_dict[feature][0], param, feature_dict[feature][1], stdev_0, stdev_1, SEM_0, SEM_1]
			mean_list.append([feature_new, mean_0, mean_1, median_0, median_1, fgroup, feature_dict[feature][0], param, feature_dict[feature][1], stdev_0, stdev_1, SEM_0, SEM_1])


	mean_df =pd.DataFrame(mean_list, columns=mean_header)
	
	return mean_dict, mean_df


def Calc_Feature_Importance(data, feature_dict, attribute_cols, numfeature):
	colnames = list(feature_dict.keys())
	colnames_orig = colnames
	if attribute_cols[7] not in colnames:
		colnames.append(attribute_cols[7])
	
	data = data[colnames]
	#data['EVERHOSP'].astype('int')
	data = data.astype({"EVERHOSP": int})

	X, y, featurenames = load_dataset(data)
	#model = RandomForestClassifier(n_estimators=10, n_jobs=8)
	#scores = cross_val_score(model, X, y, cv=5)

	feature_imp_dict = {}
	for i in range(10):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		model = XGBRegressor(n_estimators=100)
		model.fit(X_train, y_train)
		featimp = model.feature_importances_
		#input("")
	

		for i in range(len(featimp)):
			if colnames_orig[i] not in feature_imp_dict:
				feature_imp_dict[colnames_orig[i]] = []
			feature_imp_dict[colnames_orig[i]].append(featimp[i])
	
	feature_imp_dict_mean = {}
	for k in feature_imp_dict:
		k1 = re.sub(",", " ", k)
		mean_val = stat.mean(feature_imp_dict[k])
		feature_imp_dict_mean[k] = [mean_val]

	feature_imp_dict_mean = dict(sorted(feature_imp_dict_mean.items(), key=lambda item: item[1][0], reverse=True))
	feature_bar_plot(feature_imp_dict_mean, "metabolite-1-featureimp-xgboost-all-mean", "metabolite-1-featureimp-xgboost-all-mean")
	#exit()
	top10feat = list(feature_imp_dict_mean.keys())[0:numfeature]
	
	top10feat.append("EVERHOSP")

	data_new = data[top10feat]

	return data_new
def random_forest_classification(newdata):

	# Generating a random dataset for demonstration purposes
	X, y, featurenames = load_dataset(newdata)
	# Splitting the data into training and testing sets
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

	# Creating a Random Forest classifier object and fitting the model on the training data
	clf = RandomForestClassifier(n_estimators=250, random_state=42)

	# balance the training set using undersampling
	#from imblearn.under_sampling import RandomUnderSampler
	#undersampler = RandomUnderSampler(random_state=42)

	
	scores = cross_val_score(clf, X, y, cv=5)
	

	y_pred = cross_val_predict(clf, X, y, cv=5)
	
	tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

	sensitivity, specificity, precision, recall = calculate_metrics(tp, fp, tn, fn)

	return scores, sensitivity, specificity, precision, recall
	
def calculate_metrics(tp, fp, tn, fn):
	sensitivity = tp / (tp + fn)
	specificity = tn / (tn + fp)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	return sensitivity, specificity, precision, recall


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_tSNE(data, figname):
	X, y, featurenames = load_dataset(data)

	# Run t-SNE
	tsne = TSNE(n_components=2, random_state=42, perplexity=30)
	X_tsne = tsne.fit_transform(X)

	# Plot the results
	plt.figure(figsize=(10, 8))
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
	plt.colorbar(ticks=range(10))
	plt.title('t-SNE')
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.savefig("images_out/tsne-{}.png".format(figname))

	n_clusters = 5
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	cluster_labels = kmeans.fit_predict(X_tsne)
	silhouette_avg = silhouette_score(X_tsne, cluster_labels)
	print(f"Silhouette Score: {silhouette_avg}")

	sample_silhouette_values = silhouette_samples(X_tsne, cluster_labels)
	plt.figure(figsize=(8, 6))
	y_lower = 10
	for i in range(n_clusters):
		# Aggregate the silhouette scores for samples belonging to cluster i
		ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
		ith_cluster_silhouette_values.sort()

		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i

		color = plt.cm.viridis(float(i) / n_clusters)
		plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

		# Label the silhouette plots with their cluster numbers at the middle
		plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

		# Compute the new y_lower for the next plot
		y_lower = y_upper + 10  # 10 for the 0 samples

	plt.title("Silhouette Plot for t-SNE Clustering")
	plt.xlabel("Silhouette Coefficient Values")
	plt.ylabel("Cluster Label")

	# The vertical line represents the average silhouette score
	plt.axvline(x=silhouette_avg, color="red", linestyle="--")

	plt.savefig("images_out/silhoutte-{}.png".format(figname))

def plotPCA(data, figname, target_colname):
	data = scale_data(data, target_colname)

	X, y, featurenames = load_dataset(data)
	#pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5, degree=5)
	pca = PCA(n_components=4)
	X_pca  = pca.fit_transform(X)

	variance_ratios = pca.explained_variance_ratio_
	print("variance_ratios", variance_ratios)
	# Plot the results
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
	plt.colorbar(ticks=range(10))
	plt.title('PCA')
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.savefig("images_out/pca-{}.png".format(figname))
	df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
	df_pca[target_colname] = y
	plot_all_components(X_pca, y, target_colname)
	return df_pca

def plot_all_components(X_pca, y, param):
	fig, axes = plt.subplots(4, 4, figsize=(12, 12))

	# Plot each pair of PCA components
	for i in range(4):
		for j in range(4):
			if i == j:
				# Plot a histogram of the PCA component if it's on the diagonal
				axes[i, j].hist(X_pca[:, i], bins=10)
				axes[i, j].set_xlabel(f'PCA {i+1}')
				axes[i, j].set_ylabel('Frequency')
			else:
				# Scatter plot of the PCA components
				axes[i, j].scatter(X_pca[:, i], X_pca[:, j], c=y)
				axes[i, j].set_xlabel(f'PCA {i+1}')
				axes[i, j].set_ylabel(f'PCA {j+1}')

	# Adjust spacing and layout
	plt.tight_layout()

	# Show the plot
	plt.savefig("images_out/all_vs_all{}-pca.PNG".format(param))

def doRFE(data, param):
	X, y, featurenames = load_dataset(data)
	model = LogisticRegression(solver='lbfgs', max_iter=4000, random_state=42) #ElasticNet(alpha=0.08, l1_ratio=0.9) #RandomForestClassifier(random_state=42, n_estimators=150) #
	
	if param == "EVERHOSP":
		rfe = RFE(model, n_features_to_select=10)	#num feature: 6 for blind testing
	elif param == "LONGCOVID3":
		rfe = RFE(model, n_features_to_select=10)	#num feature: 8 for blind testing
	
	fit = rfe.fit(X, y)
	bool_list = fit.support_
	bool_list = np.append(bool_list, True)				#For Target
	selected_cols = data.columns[bool_list]
	data_selected = data[selected_cols]


	return data_selected, data_selected.columns.tolist()

def kendall_corr(data):
	allcol = data.columns.tolist()
	target_col = allcol[-1]
	target = data[allcol[-1]].tolist()
	del allcol[-1]
	new_col = []
	for col in allcol:
		coldata = data[col].tolist()

		corr, pval = stats.kendalltau(coldata, target)
		if pval < 0.05:
			new_col.append(col)

	new_col.append(target_col)
	newdata = data[new_col]

	return newdata



def format_names(inpdict):	
	outdict = {}
	for k in inpdict:
		k1 = re.sub("#\d+$", "", k)
		outdict[k1] =inpdict[k]
	return outdict

def get_common_metabolite(inpdict):
	met_1 = inpdict["METABOLITES-1"]
	met_2 = inpdict["METABOLITES-2"]
	ratio_1 = inpdict["RATIOS-1"]
	ratio_2 = inpdict["RATIOS-2"]
	drugs_1 = inpdict["DRUGS-1"]
	drugs_2 = inpdict["DRUGS-2"]

	met_1 = format_names(met_1)
	met_2 = format_names(met_2)
	ratio_1 = format_names(ratio_1)
	ratio_2 = format_names(ratio_2)
	drugs_1 = format_names(drugs_1)
	drugs_2 = format_names(drugs_2)
		
	met_common_keys = set(met_1.keys()) & set(met_2.keys())
	ratio_common_keys = set(ratio_1.keys()) & set(ratio_2.keys())
	drugs_common_keys = set(drugs_1.keys()) & set(drugs_2.keys())
	
	return list(met_common_keys), list(ratio_common_keys), list(drugs_common_keys)


def vertical_scatter_plot(data, title, figname, rfe_list):
	rfe_list = [re.sub("#2$", "", elm) for elm in rfe_list]

	data["MeanDiff"] = data["MeanDiff"].astype(float)

	data = data.sort_values(by='MeanDiff', ascending=True)

	values = data["MeanDiff"].values.tolist()
	values1 = data["SCORE"].values.tolist()
	values1 = [60 for v in values1]
 
	#color = [color_dict[elm] for elm in data["FEATURE_GROUP"].values.tolist()]
	color = []
	for val in values:
		if val <= 0:
			color.append("darkgreen")
		else:
			color.append("darkred")

	fig, ax = plt.subplots(figsize=(10, 10))

	# Set the middle line at zero
	ax.axvline(0, color='black', linestyle='--')

	# Plot the dots
	data['Pvalue'] = data['Pvalue'].round(3)
	data['Pvalue'] = data['Pvalue'].astype(str)
	data["Ylabel"] = data['NAME'] + "(" + data['Pvalue'] + ")"

	ax.scatter(data['MeanDiff'], data['Ylabel'], s=values1, color=color)

	names = data["Ylabel"].values.tolist()

	labels = ax.get_yticklabels()
	ticks = ax.get_yticks()
	for label, tick in zip(labels, ticks):
		
		label_txt = label.get_text()
		matches = [s for s in rfe_list if re.search(s, label_txt)]
		if  matches:
			label.set_color('SaddleBrown')
	# Set the x-axis limits
	min_value = min(values)
	max_value = max(values)
	ax.set_xlim(min_value-1.5, max_value+1.5)

	# Set the title and labels
	ax.set_title(title)
	ax.set_xlabel('Mean Difference (Mean-1 - Mean-0)')
	ax.set_ylabel('Names')
	plt.tight_layout()
	# Display the plot
	plt.savefig("images_out/verticalscatter"+figname+".pdf")
	plt.savefig("images_out/verticalscatter"+figname+".png")
	plt.close()

def plot_feature_corr(df, param):
	corr = df.corr()
	# visualise the data with seaborn
	mask = np.triu(np.ones_like(corr, dtype=bool))
	sns.set_style(style = 'white')
	f, ax = plt.subplots(figsize=(11, 9))
	cmap = sns.diverging_palette(10, 250, as_cmap=True)
	sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
	plt.savefig("images_out/rfe-"+param+"_corr.png")
	plt.close()



def featuretool_processing(data, param):
	es = ft.EntitySet(id = 'data')
	es.add_dataframe(dataframe_name = 'data', dataframe = data, logical_types ={param: Categorical})
	feature_matrix, feature_names = ft.dfs(entityset=es,
	max_depth = 2, 
	verbose = 1, 
	n_jobs = 3)
	#print("featuretools")
	#print(feature_matrix.columns)
	#print(feature_names)


def get_models():
	models = dict()
	for i in range(2, 10):
		
		rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=i)
		model = RandomForestClassifier()
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

def plotRFE(df):

	# define dataset
	X, y, featurenames = load_dataset(df)
	# get the models to evaluate
	models = get_models()
	# evaluate the models and store results
	results, names = list(), list()
	for name, model in models.items():
		scores = evaluate_model(model, X, y)
		results.append(scores)
		names.append(name)
		print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
	# plot model performance for comparison
	plt.boxplot(results, labels=names, showmeans=True)
	plt.savefig("images_out/rfe_eval.PNG")


def oversampling(inpdf, param):
	
	X = inpdf.drop(param, axis=1)  #
	y = inpdf[param]
	smote = SMOTE(sampling_strategy='auto', random_state=42)
	X_resampled, y_resampled = smote.fit_resample(X, y)

	resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
	resampled_data[param] = y_resampled

	return resampled_data

def search_hyperparam(ml_classifier, X_train, y_train):
	
	param_grid = {
	'n_estimators': [50, 100, 150, 250, 300, 400],
	'max_depth': [None, 10, 20, 30, 50, 100],
	'min_samples_split': [2, 5, 10, 15, 20, 25],
	'min_samples_leaf': [1, 2, 4, 8, 10, 12]
	}
	
	grid_search = GridSearchCV(ml_classifier, param_grid, cv=5, n_jobs=-1)
	grid_search.fit(X_train, y_train)
	
	print("Best Hyperparameters:", grid_search.best_params_)

def ML_model(data, title, param):

	class_counts = data[param].value_counts()

	X, y, featurenames = load_dataset(data)

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	
	# Create a Random Forest classifier
	ml_classifier = RandomForestClassifier(random_state=42, min_samples_leaf=1, min_samples_split=5, n_estimators=150)	#xgb.XGBClassifier()
	ml_classifier_tosave = RandomForestClassifier(random_state=42, min_samples_leaf=1, min_samples_split=5, n_estimators=150)	#xgb.XGBClassifier()
	
	# Train the classifier
	ml_classifier.fit(X_train, y_train)
	
	#Find Best Hyper Parameters
	search_hyperparam(ml_classifier, X_train, y_train)
	
	ml_classifier_tosave.fit(X, y)
	model_filename = 'images_out/trained_RF_model.joblib'
	joblib.dump(ml_classifier_tosave, model_filename)
	# Make predictions
	y_pred = ml_classifier.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)

	print(f"Accuracy Score: {accuracy:.2f}")

	# Predict probabilities for the test set
	y_pred_prob = ml_classifier.predict_proba(X_test)[:, 1]

	# Compute ROC curve and ROC AUC score
	fpr, tpr, threshold = roc_curve(y_test, y_pred_prob, drop_intermediate=False)

	roc_auc = roc_auc_score(y_test, y_pred_prob)

	# Compute precision-recall curve and AUC score
	precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
	prc_auc = auc(recall, precision)
	#5-Fold CrossValidation Test
	do_5FoldCrossvalidation(X, y, ml_classifier, title)
	
	#Estimate Benchmarking Parameters
	conf_matrix = confusion_matrix(y_test, y_pred)
	sensitivity = recall_score(y_test, y_pred)
	specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
	accuracy = accuracy_score(y_test, y_pred)
	precisionsc = precision_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	return accuracy, roc_auc, prc_auc, fpr, tpr, recall, precision, model_filename, sensitivity, specificity, precisionsc, f1

def create_confusionmatrix(actual_labels, predicted_labels):
	data = {'Actual': actual_labels, 'Predicted': predicted_labels}
	df = pd.DataFrame(data)

	# Create a confusion matrix
	confusion = confusion_matrix(df['Actual'], df['Predicted'])
	return confusion

def testML_realdata(modelname, featureset, param, testdata, case_name):
	testdata = testdata[featureset]
	patient_id_list = testdata.index.values.tolist()
	testdata_log = df_log_transform(testdata, param)
	testdata_log_scaled = scale_data(testdata_log, param)	

	X_test, y_test, featurenames = load_dataset(testdata_log_scaled)
	loaded_model = joblib.load(modelname)
	y_pred = loaded_model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	y_prob = loaded_model.predict_proba(X_test)[:, 1]
	#FPR: 19, TPR: 90, Threshold: 0.528
	is_misclassified = [label != (prob >= 0.5) for label, prob in zip(y_test, y_prob)]

	data_bplot = {
    	"Label": y_test,
    	"Score": y_prob,
		"is_misclassified": is_misclassified
	}

	df_bplot = pd.DataFrame(data_bplot)
	df_bplot.to_csv(f"images_out/DF-value-{param}-{case_name}.csv", index=False)
	create_label_boxplot(df_bplot, param)
	#exit()
	
	conf_matrix = confusion_matrix(y_test, y_pred)
	sensitivity = recall_score(y_test, y_pred)
	specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	
	print("Sensitivity (Recall): {:.2f}".format(sensitivity*100))
	print("Specificity: {:.2f}".format(specificity*100))
	print("Accuracy: {:.2f}".format(accuracy*100))
	print("Precision: {:.2f}".format(precision*100))
	print("F1 Score: {:.2f}".format(f1*100))

	contingency_table  = create_confusionmatrix(y_test, y_pred)

	odds_ratio2, p_value2 = stats.fisher_exact(contingency_table)
	print(f"Fischer exact Test {param}", p_value2)
 

	# Compute ROC curve and ROC AUC score
	fpr_list, tpr_list, threshold = roc_curve(y_test, y_prob, drop_intermediate=False)
	#print(fpr, tpr, threshold)
	roc_auc = roc_auc_score(y_test, y_prob)
	print(f"ROC-AUC Score from Real Data: {roc_auc}")
	plt.figure()
	plt.plot(fpr_list, tpr_list, color='black', lw=2, label='ROC curve (Area = {:.2f})'.format(roc_auc))	#, color='blue', label='ROC curve (AUC = {:.2f}, ACC = {:.2f})'.format(roc_auc, avg_accuracy))
	plt.plot([0, 1], [0, 1], color='black', linestyle='--')
	plt.xlabel('1-Specifcity')
	plt.ylabel('Sensitivity')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc='lower right')
	plt.savefig(f"images_out/real-data-roc-{case_name}-{param}.pdf", dpi=360)
	plt.close()
	plt.figure()

	"""
	new_threshold = 0
	new_threshold_everhosp = 0.5
	new_threshold_longcovid3 = 0.4
	if param == "EVERHOSP":
		new_threshold = new_threshold_everhosp
	elif param == "LONGCOVID3":
		new_threshold = new_threshold_longcovid3
	
	
	y_pred = np.where(y_prob >= new_threshold, 1, 0)
	accuracy = accuracy_score(y_test, y_pred)
	for i in range(len(y_test)):
		print(y_test[i], y_pred[i])
	
	print(f"Accuracy with New Threshold {param} {new_threshold}: {accuracy:.2f}")
	#for i in range(len(patient_id_list)):
	#	print(f"{param}, {patient_id_list[i]}, {int(y_test[i])}, {y_pred[i]}")

	contingency_table  = create_confusionmatrix(y_test, y_pred)
	odds_ratio2, p_value2 = stats.fisher_exact(contingency_table)
	print(f"Fischer exact Test {param}", p_value2)
	
	for new_threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
		y_prob = loaded_model.predict_proba(X_test)[:, 1]
		y_pred = np.where(y_prob >= new_threshold, 1, 0)
		accuracy = accuracy_score(y_test, y_pred)
		print(y_test)
		print(y_pred)
		print(f"Accuracy with New Threshold {param} {new_threshold}: {accuracy:.2f}")
	exit()
	"""
def	create_label_boxplot(df, param):
	marker_shapes = {True: 's', False: 'o'}
	palette_colors = {0: 'black', 1: (0/255, 197/255, 205/255)}
	x_order = [0, 1]
	#sns.set(style="whitegrid")
	plt.figure(figsize=(6, 4))

	sns.boxplot(x="Label", y="Score", data=df, palette="Set2", width=0.3,
                  boxprops=dict(facecolor='none'))
	#sns.swarmplot(x="Label", y="Score", hue="is_misclassified", data=df, 
    #           marker='x', palette={True: 'red', False: 'blue'}, size=8, ax=ax)
	#sns.stripplot(data=df, x="Label", y="Score")
	misclassified_data = df[df['is_misclassified']]
	correctly_classified_data = df[~df['is_misclassified']]

	sns.stripplot(x="Label", y="Score", data=correctly_classified_data, hue="Label",
              marker=marker_shapes[False], dodge=False, edgecolor='gray', size=8, legend=False, palette=palette_colors)
	sns.stripplot(x="Label", y="Score", data=misclassified_data, hue="Label",
              marker=marker_shapes[True], dodge=False, edgecolor='gray', size=8, legend=False, palette=palette_colors)
	plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
	plt.ylim(0, 1)
	plt.yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.5", "0.6", "0.8", "1"])
	plt.xlabel("Label")
	plt.ylabel("Prediction Probability Score")
	#plt.title("Box Plot with Separate Markers for Each Label Class")
	plt.savefig(f"images_out/boxplot-classification-{param}.png")
	plt.close()
	#exit()
	# Calculate ROC curve
	fpr, tpr, thresholds = roc_curve(df['Label'], df['Score'])
	roc_auc = auc(fpr, tpr)

	# Plot ROC curve
	plt.figure(figsize=(8, 6))
	plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (Area = {:.2f})'.format(roc_auc))
	plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc='lower right')
	plt.savefig(f"images_out/roc-alldata-{param}.pdf", dpi=300)
	plt.close()

def do_5FoldCrossvalidation(X, y, model, title):
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	# Initialize subplots for ROC and PRC curves
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

	# Iterate over each fold
	for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		# Train the model
		model.fit(X_train, y_train)

		# Get predicted probabilities
		y_score = model.predict_proba(X_test)[:, 1]

		# Calculate ROC curve and AUC
		fpr, tpr, _ = roc_curve(y_test, y_score)
		roc_auc = auc(fpr, tpr)

		# Calculate PRC curve and AP
		precision, recall, _ = precision_recall_curve(y_test, y_score)
		prc_auc = average_precision_score(y_test, y_score)

		# Plot ROC curve
		axes.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc:.2f})')

		# Plot PRC curve
		#axes[1].plot(recall, precision, label=f'Fold {fold} (AP = {prc_auc:.2f})')

	# Set titles and labels
	axes.set_title('Receiver Operating Characteristic (ROC) Curves')
	axes.set_xlabel('Sensitivity')
	axes.set_ylabel('1-Specificity')

	#axes[1].set_title('Precision-Recall Curves')
	#axes[1].set_xlabel('Recall')
	#axes[1].set_ylabel('Precision')

	# Add legend
	axes.legend(loc='lower right')
	#axes[1].legend(loc='lower left')

	# Display the plots
	plt.tight_layout()
	plt.savefig(f"images_out/5fold-CV-{title}.pdf")


def plot_roc_prc_curve(fpr_list, tpr_list, recall_list, precision_list, roc_auc, prc_auc, exp_type, title, avg_accuracy, cvtype):
	plt.figure()
	for i in range(len(fpr_list)):
		fpr = fpr_list[i]
		tpr = tpr_list[i]

		plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (Area = {:.2f})'.format(roc_auc))	#, color='blue', label='ROC curve (AUC = {:.2f}, ACC = {:.2f})'.format(roc_auc, avg_accuracy))
		plt.plot([0, 1], [0, 1], color='black', linestyle='--')
	
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc='lower right')
	plt.savefig(f"images_out/ROC-{exp_type}-{title}-{cvtype}.pdf", dpi=360)
	plt.close()
	plt.figure()
	# Plot Precision-Recall curve
	for i in range(len(recall_list)):
		recall = recall_list[i]
		precision = precision_list[i]
		plt.plot(recall, precision)	#, color='red', label='PRC curve (AUC = {:.2f}, ACC = {:.2f})'.format(prc_auc, avg_accuracy))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	#plt.legend(loc='lower left')
	plt.savefig(f"images_out/PRC-{exp_type}-{title}-{cvtype}.png")
	plt.close()

def versiontuple(v):
	return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
						   np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
					alpha=0.8, c=cmap(idx),
					marker=markers[idx], label=cl)

	# highlight test samples
	if test_idx:
		# plot all samples
		if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
			X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
			warnings.warn('Please update to NumPy 1.9.0 or newer')
		else:
			X_test, y_test = X[test_idx, :], y[test_idx]

		plt.scatter(X_test[:, 0],
					X_test[:, 1],
					c='',
					alpha=1.0,
					linewidths=1,
					marker='o',
					s=55, label='test set')


def plot_decision_boundary(X, y, model):
	model.fit(X,y)
	# Set the minimum and maximum values for the features
	min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
	min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
	# define the x and y scale
	x1grid = np.arange(min1, max1, 0.1)
	x2grid = np.arange(min2, max2, 0.1)	

	xx, yy = np.meshgrid(x1grid, x2grid)
	# Predict the class labels for the grid points
	r1, r2 = xx.flatten(), yy.flatten()
	r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
	grid = np.hstack((r1,r2))
	yhat = model.predict(grid)
	zz = yhat.reshape(xx.shape)
	# Plot the decision boundary
	plt.contourf(xx, yy, zz, cmap='Paired')

	for class_value in range(2):
		# get row indexes for samples with this class
		row_ix = np.where(y == class_value)
		# create scatter of these samples
		plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')

	#plt.xlabel('Feature 1')
	#plt.ylabel('Feature 2')
	#plt.title('Decision Boundary')
	plt.savefig("decision_boundary.png")
	plt.close()

def downsample_data(dfinp, param, int_num):

	majority_class = dfinp[dfinp[param] == 0]
	minority_class = dfinp[dfinp[param] == 1]
	#print(majority_class.shape)
	# Downsample the majority class
	downsampled_majority = resample(majority_class,
								replace=False,  # sampling without replacement
								n_samples=len(minority_class),  # match minority class size
								random_state=int_num)  # for reproducibility

	# Combine the downsampled majority class with the minority class
	#print("Majority", downsampled_majority.index.tolist(), len(downsampled_majority.index.tolist()))
	#print("Minority", minority_class.index.tolist(), len(minority_class.index.tolist()))
	balanced_dataset = pd.concat([downsampled_majority, minority_class])

	# Shuffle the dataset
	balanced_dataset = balanced_dataset.sample(frac=1)
	return balanced_dataset


def impute_higher_and_lower(coldata):
	trimd_data = []
	higher_ind = []
	lower_ind = []
	xdata = []
	#print(coldata)
	for i in range(len(coldata)):
		if re.search("HIGHER", str(coldata[i]), flags=re.IGNORECASE):
			higher_ind.append(i)
		elif re.search("LOWER", str(coldata[i]), flags=re.IGNORECASE):
			lower_ind.append(i)
		elif str(coldata[i]) != 'nan':							#IGNORING Missing Values in Calculating Mean
			trimd_data.append(coldata[i])
	trimd_data = [float(elm) for elm in trimd_data]
	stdev = stat.stdev(trimd_data)
	maxv = max(trimd_data)
	minv = min(trimd_data)
	imputed_max = maxv + 2*stdev
	imputed_min = minv - 2*stdev
	
	if imputed_min <0 and minv > 0:
		imputed_min = 0
	
	for i in range(len(coldata)):
		if i in higher_ind:
			coldata[i] =  imputed_max
		elif i in lower_ind:
			coldata[i] =  imputed_min
		xdata.append(i)
	
	#print("Min Imputed 1", imputed_min, minv, stdev)
	#print("Min Imputed 2", imputed_min, minv, stdev)
	
	coldata = [float(elm) for elm in coldata]
	return coldata

def impute_outliers(data, attribute_cols):
	###CLEANING THE DATASET
	#REMOVE THE WORDS HIGHER & LOWER
	null_sum = {}
	columns  = data.columns.to_list()
	for col in columns:		   
		if col in attribute_cols:
			continue
		null_sum[col] = data[col].isnull().sum()
		coldata = data[col].tolist()
		if 'HIGHER' in coldata or 'LOWER' in coldata:
			#print(col)
			#print(coldata)
			coldata = impute_higher_and_lower(coldata)
			data[col] = coldata   
	null_sum = dict(sorted(null_sum.items(), key=lambda x:x[1], reverse=True))
	return data, null_sum


def drawboxplot(dfinp, title, param):
	df = dfinp

	# Get the number of columns
	num_columns = len(df.columns)

	# Calculate the number of rows and columns for subplots
	num_rows = (num_columns + 2) // 4
	num_cols = 4

	# Set up subplots
	fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(14, 4 * num_rows))

	# Flatten the axes array for easier iteration
	axes = axes.flatten()
	colors = {0: 'blue', 1: 'orange'}
	# Loop through the columns and draw boxplots in subplots
	for i, column in enumerate(df.columns[:-1]):
		box_data = [df[column][df[param] == 0], df[column][df[param] == 1]]
		bp = axes[i].boxplot(box_data, labels=['0', '1'], patch_artist=True, widths=0.7)
	
		for patch, color in zip(bp['boxes'], [colors[0], colors[1]]):
			patch.set_facecolor(color)
	
		#axes[i].set_title(f'Boxplot for {column} by Target')
		column_f = re.sub("#2$", "", column)
		axes[i].set_xlabel(param)
		axes[i].set_ylabel(column_f)
	# Adjust layout
	plt.tight_layout()

	# Show the plots
	plt.savefig(f"images_out/boxplot-{title}.png")



def vertical_scatter_plot_errorbar(data, title, figname, rfe_list):
	rfe_list = [re.sub("#2$", "", elm) for elm in rfe_list]

	data["MeanDiff"] = data["MeanDiff"].astype(float)

	data = data.sort_values(by='MeanDiff', ascending=True)
	specific_strings = ["Q/T", "K/T"]
	mask = data['NAME'].isin(specific_strings)
	data = pd.concat([ data[~mask], data[mask]])
	values = data["MeanDiff"].values.tolist()
	values1 = data["SCORE"].values.tolist()
	values1 = [120 for v in values1]
 
	#color = [color_dict[elm] for elm in data["FEATURE_GROUP"].values.tolist()]
	color = []
	for val in values:
		if val <= 0:
			color.append("#0095FF")
		else:
			color.append("#FF2600")

	fig, ax = plt.subplots(figsize=(18, 18))

	# Set the middle line at zero
	ax.axvline(0, color='black', linestyle='--')

	# Plot the dots
	#print(data.head())
	#exit()
	data["Ylabel"] = data['NAME']	# + "(" + data['Pvalue'] + ")"

	for i, row in data.iterrows():
		dot_color = '#0095FF' if row['MeanDiff'] < 0 else '#FF2600'
		ax.errorbar(row['MeanDiff'], row['Ylabel'], xerr=row['SME-0'], fmt='o', color=dot_color, ecolor="black", capsize=3, elinewidth=1, markersize=12)
		#ax.errorbar(row['MeanDiff'], row['Ylabel'], fmt='o', color=color[i], ecolor="black", capsize=5)
	
	#ax.errorbar(data['MeanDiff'], data['Ylabel'], xerr=SEM_group_0, fmt='o', color="black", capsize=4, elinewidth=1)
	#ax.scatter(data['MeanDiff'], data['Ylabel'], s=values1, color=color, alpha=1)
	names = data["Ylabel"].values.tolist()
	ax.tick_params(axis='both', labelsize=20)
	labels = ax.get_yticklabels()
	ticks = ax.get_yticks()
	for label, tick in zip(labels, ticks):
		
		label_txt = label.get_text()
		matches = [s for s in rfe_list if re.search(s, label_txt)]
		#if  matches:
		#	label.set_color('SaddleBrown')
	# Set the x-axis limits
	min_value = min(values)
	max_value = max(values)
	#ax.set_xlim(15, 30)

	# Set the title and labels
	ax.set_title(title)
	ax.set_xlabel('Mean Difference (Mean-1 - Mean-0)', fontsize=20)
	ax.set_ylabel('Names', fontsize=20)
	plt.tight_layout()
	# Display the plot
	plt.savefig("images_out/verticalscatter_errbar"+figname+".pdf", dpi=300)
	plt.savefig("images_out/verticalscatter_errbar"+figname+".png")
	plt.close()
	print(figname)
	#exit()

from matplotlib.lines import Line2D
def bar_plot_errorbar(data, title, figname, rfe_list):
	rfe_list = [re.sub("#2$", "", elm) for elm in rfe_list]

	data["MeanDiff"] = data["MeanDiff"].astype(float)

	data = data.sort_values(by='MeanDiff', ascending=True)

	values = data["MeanDiff"].values.tolist()
	values1 = data["SCORE"].values.tolist()
	values1 = [60 for v in values1]
 
	#color = [color_dict[elm] for elm in data["FEATURE_GROUP"].values.tolist()]
	color = []
	for val in values:
		if val <= 0:
			color.append("darkgreen")
		else:
			color.append("darkred")

	fig, ax = plt.subplots(figsize=(21, 18))

	# Set the middle line at zero
	ax.axvline(0, color='black', linestyle='--')

	# Plot the dots
	#data['Pvalue'] = data['Pvalue'].round(3)
	#data['Pvalue'] = data['Pvalue'].astype(str)
	data["Ylabel"] = data['NAME']	# + "(" + data['Pvalue'] + ")"
	#print(data.head(10))
	colors = ['darkred' if v1 > v2 else 'darkgreen' for v1, v2 in zip(data['MEAN-1'], data['MEAN-0'])]
	bar_height = 0.4

	bars1 =ax.barh(data['NAME'], data['MEAN-1'], xerr=data['STDev-1'], capsize=5, color=colors, edgecolor='black', label='Value 1', height=bar_height)
	bars2 = ax.barh(data['NAME'], -data['MEAN-0'], xerr=data['STDev-0'], capsize=5, color=colors, edgecolor='black', label='Value 2', height=bar_height)
	names = data["Ylabel"].values.tolist()

	labels = ax.get_yticklabels()
	ticks = ax.get_yticks()

	# Set the title and labels
	ax.set_title(title)
	#ax.set_xlabel('Mean Difference (Mean-1 - Mean-0)')
	ax.set_ylabel('NAMES')
	for bar1, bar2 in zip(bars1, bars2):
		plt.text(bar1.get_width()/2, bar1.get_y() + bar_height / 2, str(round(abs(bar1.get_width()),2)), ha='center', va='center', color='white')
		plt.text(bar2.get_width()/2, bar2.get_y() + bar_height / 2, str(round(abs(bar2.get_width()),2)), ha='center', va='center', color='white')
		#plt.tight_layout()
	# Display the plot
	plt.xticks([])
	# Create a color legend
	legend_elements = [
		Line2D([0], [0], color='darkgreen', lw=8, label='Upregulated'),
		Line2D([0], [0], color='darkred', lw=8, label='Downregulated ')
	]

	plt.legend(handles=legend_elements, title='Color Legend', loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
	#plt.legend()
	plt.savefig("images_out/verticalscatter_errbar"+figname+".pdf", dpi=360)
	#plt.savefig("images_out/verticalscatter_errbar"+figname+".png")
	plt.close()
