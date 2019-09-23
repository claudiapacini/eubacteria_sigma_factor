import pandas as pd
import tensorflow as tf
import numpy as np

def main():
    X, y = get_word_embeddings()
    print('finished training word emeddings')
    # use the seuence embeddings to predict the target class
    X.head()


def get_word_embeddings():
    X, y = load_input_data()
    X.head()
    y.head()
    print("finished loading input data")

    df, dictionary = get_word_embedding_input_data(X.word_embedding_features.values)
    df.head()
    print("finished creating word embeddings training data from input")

    word_vectors = train(df.X, df.y, dictionary)
    print(word_vectors)

    vector_dictionary = get_vector_dictionary(dictionary, word_vectors)
    print(vector_dictionary)

    X["tokens_flat"] = X.ngram_tokens.map(lambda x: flatten(x))
    X["token_vectors"] = X.tokens_flat.map(lambda x: vector_dictionary[x])
    X["token_vectors_flat"] = X.token_vectors.map(lambda x: flatten(x))

    return X, y


def load_input_data():
    X = pd.read_csv(
        "./PromoterTrain.csv", header=0, names=["id", "sequence"], index_col="id", sep=","
    )
    X.head()

    y = pd.read_csv("./SigmaTrain.csv", header=0, index_col="id", sep=",")
    y.head()
    
    X["ngram_tokens"] = X.sequence.map(lambda x: splitter(x))
    X["word_embedding_features"] = X.ngram_tokens.map(
        lambda x: word_embedding_feature_pairs(x)
    )
    return X, y


def get_vector_dictionary(dictionary, word_vectors):
    vector_dictionary = {}
    for k, v in enumerate(dictionary):
        vector_dictionary[k] = word_vectors[v]
    return vector_dictionary


def flatten(arr):
    res = []
    for i in arr:
        res += i
    return res


def splitter(a: str, n=6):
    results = []
    for start in range(len(a)):
        res = []
        for i in range(start, len(a), n):
            val = a[i : i + n]
            # only append if it contains the full n chars
            if len(val) == n:
                res.append(val)
        if len(res) > 1:
            results.append(res)
    return results


def word_embedding_feature_pairs(corpus, window_size=1):
    pairs = []
    for doc in corpus:
        for idx, seq in enumerate(doc):
            for neighbour in doc[
                max(idx - window_size, 0) : min(idx + window_size, len(doc)) + 1
            ]:
                if neighbour != seq:
                    pairs.append((seq, neighbour))
    return pairs




def get_word_embedding_input_data(nested_lists):
    unique_seqs = set()
    x = []
    y = []

    for i in nested_lists:
        for a, b in i:
            x.append(a)
            y.append(b)
            unique_seqs.add(a)
            unique_seqs.add(b)

    dictionary = {}
    for i, seq in enumerate(unique_seqs):
        dictionary[seq] = i

    df = pd.DataFrame({"X": x, "y": y})
    return df, dictionary


def one_hot_encode_seq(data_point_index, one_hot_dim):
    res = np.zeros(one_hot_dim)
    res[data_point_index] = 1
    return res


def train(X_strings, y_strings, dictionary, hidden_layer_dim=6):
    input_dim = len(dictionary)

    X_encoded = []  # input seq as one hot encoed
    Y_encoded = []  # target seq as . one hot encoded

    for seq, target in zip(X_strings, y_strings):
        X_encoded.append(one_hot_encode_seq(dictionary[seq], input_dim))
        Y_encoded.append(one_hot_encode_seq(dictionary[target], input_dim))

    X_train = np.asarray(X_encoded)
    y_train = np.asarray(Y_encoded)

    model = tf.keras.Sequential(
        [
            # input layer
            tf.keras.layers.Dense(input_dim, activation="relu"),
            # hidden layer
            tf.keras.layers.Dense(hidden_layer_dim, activation="relu"),
            # output layer
            tf.keras.layers.Dense(input_dim, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.train.AdamOptimizer(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(X_train, y_train, epochs=5)
    # extract the embeddings
    return model.layers[1].get_weights()[0]



if __name__ == "__main__":
    main()