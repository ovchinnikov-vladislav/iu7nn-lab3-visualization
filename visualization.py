from matplotlib import pyplot
import seaborn
import sklearn.preprocessing as skl_prep
import pandas
import umap
from sklearn import manifold
from sklearn import decomposition

dataframe = pandas.read_csv("datasets/voice-train.csv")
strings = dataframe.head()
print(strings)
quality = dataframe.groupby("label")["label"].count()
print(dataframe.shape)
print(quality)

pyplot.bar(quality.index, quality.values)
pyplot.savefig("images/quality.png")
pyplot.clf()
seaborn_pairplot = seaborn.pairplot(dataframe, hue="label",
                            vars=["meanfreq", "meanfun", "median", "minfun", "maxfun", "meandom", "mindom", "maxdom"])
seaborn_pairplot.savefig("images/pairplot.png")

label = "label"
columns = []
for col in dataframe.columns:
    if col != label:
        columns.append(col)

print(columns)

dataframe.loc[dataframe['label'] == 'male', 'label'] = 0
dataframe.loc[dataframe['label'] == 'female', 'label'] = 1

# Нормализация
x = dataframe[columns].values
min_max_scaler = skl_prep.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_dataframe = pandas.DataFrame(x_scaled, columns=columns)
norm_dataframe[label] = dataframe[label]

strings = norm_dataframe.head()
print(strings)

pyplot.clf()

pca = decomposition.PCA(n_components=2)
pca.fit(norm_dataframe[columns])
pca_data = pca.transform(norm_dataframe[columns])

pyplot.scatter(pca_data[:, 0], pca_data[:, 1], c=norm_dataframe[label])
pyplot.savefig("images/pca.png")

pyplot.clf()

pca = decomposition.PCA(n_components=3)
pca.fit(norm_dataframe[columns])
pca_data = pca.transform(norm_dataframe[columns])

pyplot.clf()

tsne_data = manifold.TSNE(n_components=2, random_state=0, learning_rate=100, perplexity=20).fit_transform(
    norm_dataframe[columns])
pyplot.scatter(tsne_data[:, 0], tsne_data[:, 1], c=norm_dataframe[label])
pyplot.savefig("images/tsne.png")

pyplot.clf()

pyplot.clf()

reducer = umap.UMAP()

embedding = reducer.fit_transform(norm_dataframe[columns])
pyplot.scatter(embedding[:, 0], embedding[:, 1], c=norm_dataframe[label])
pyplot.savefig("images/umap.png")
