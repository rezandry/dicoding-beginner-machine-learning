from sklearn import preprocessing
data = [[12000000, 33], [35000000, 45], [4000000, 23], [6500000, 26], [9000000, 29]]

# Normalization
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
normalization = scaler.transform(data)


# Standarization
scaler = preprocessing.StandardScaler().fit(data)
standarization = scaler.transform(data)