from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a feature extraction process
def get_bow_ngrams_from_doc(doc, n=1, use_lemma=True, word_filter=[], pos_filter=[]):
    tokens = [token.lemma_.lower() if use_lemma else token.text.lower() for token in doc if (token.tag_ not in pos_filter) and (token.text not in word_filter)]
    tokens = list(ngrams(tokens, n=n))
    tokens = [" ".join(token) for token in tokens]
    return tokens

def get_bop_ngrams_from_doc(doc, n=1):
    tokens = [token.tag_ for token in doc]
    tokens = list(ngrams(tokens, n=n))
    tokens = [" ".join(token) for token in tokens]
    return tokens

def get_dep_triplet_from_doc(doc, use_lemma=False,triplet_type="wdw"):
    if triplet_type == "wdw":
        tokens = ["_".join([token.head.text, token.dep_, token.text]) if not use_lemma else "_".join([token.head.lemma_, token.dep_, token.lemma_]) for token in doc]
        return tokens
    elif triplet_type == "wdp":
        tokens = ["_".join([token.head.text, token.dep_, token.tag_]) if not use_lemma else "_".join([token.head.lemma_, token.dep_, token.tag_]) for token in doc]
        return tokens
    elif triplet_type == "pdw":
        tokens = ["_".join([token.head.tag_, token.dep_, token.text]) if not use_lemma else "_".join([token.head.tag_, token.dep_, token.lemma_]) for token in doc]
        return tokens
    elif triplet_type == "pdp":
        tokens = ["_".join([token.head.tag_, token.dep_, token.tag_]) for token in doc]
        return tokens

    else:
        raise ValueError(f"Invalid triplet type: {triplet_type}")

def identity_fn(x):
    return x

class BagOfFeaturesModel:
    def __init__(self, feature_type="bow", n=1, norm=None, use_idf=False, smooth_idf=False, use_lemma=True, word_filter=[], pos_filter=[], **model_args):
        self.n = n
        
        self.use_lemma = use_lemma
        self.word_filter = word_filter
        self.tag_filter = pos_filter

        self.triplet_type = feature_type.split("_")[-1] if "dep_" in feature_type else None
        self.feature_type = feature_type.split("_")[0] if "dep_" in feature_type else feature_type

        self.vectorizer = TfidfVectorizer(
            analyzer=identity_fn,
            lowercase=False,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
        )

        self.model = LogisticRegression(**model_args)

    def fit_vectorizer(self, docs):
        if self.feature_type == "bow":
            ngrams = [get_bow_ngrams_from_doc(doc, n=self.n, use_lemma=self.use_lemma, word_filter=self.word_filter) for doc in docs]
        elif self.feature_type == "bop":
            ngrams = [get_bop_ngrams_from_doc(doc, n=self.n) for doc in docs]
        elif self.feature_type == "dep":
            ngrams = [get_dep_triplet_from_doc(doc, use_lemma=self.use_lemma, triplet_type=self.triplet_type) for doc in docs]
        else:
            raise ValueError(f"Invalid feature type: {self.feature_type}")

        self.vectorizer.fit(ngrams)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def vectorize(self, docs):
        if self.feature_type == "bow":
            ngrams = [get_bow_ngrams_from_doc(doc, n=self.n, use_lemma=self.use_lemma, word_filter=self.word_filter) for doc in docs]
        elif self.feature_type == "bop":
            ngrams = [get_bop_ngrams_from_doc(doc, n=self.n) for doc in docs]
        elif self.feature_type == "dep":
            ngrams = [get_dep_triplet_from_doc(doc, use_lemma=self.use_lemma, triplet_type=self.triplet_type) for doc in docs]
        else:
            raise ValueError(f"Invalid feature type: {self.feature_type}")

        return self.vectorizer.transform(ngrams)


    def fit_model(self, X, Y):
        self.model.fit(X, Y)
        self.coefs = self.model.coef_
        self.class_list = self.model.classes_

    def predict(self, X):
        return self.model.predict(X)

    def predict_probs(self, X):
        return self.model.predict_proba(X)

    def interpret(self, x, feature_names, t=None, y=None, top_k=5):
        """
        x: numpy array, [1, n_features]
        feature_names: 1D array of strings, coming from the vectorizer $phi.feature_names$, e.g. ["feature_1", "feature_2", "feature_3"]
        t: string, e.g. "I think that the moon is made of green cheese."
        y: string, e.g. "Mind Reading"
        top_k: int, the top-k polarized attributions to show, i.e. top-k positive and top-k negative attributions
        """
        # Assert data types
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        assert isinstance(feature_names, np.ndarray), "feature_names must be a numpy array"
        assert isinstance(t, str), "t must be a string"
        assert isinstance(y, str), "y must be a string"

        # Predict the given feature vector
        y_hat = self.predict(x)[0] # string, e.g. "Mind Reading"
        y_hat_probs = self.predict_probs(x)[0] # numpy array, [n_classes,]

        # Get the index of the predicted class
        y_hat_idx = self.class_list.tolist().index(y_hat)

        # Get the coefficients for the predicted class
        y_hat_coef = self.coefs[y_hat_idx] # [n_features,]

        # Get the probability of the predicted class
        y_hat_prob = y_hat_probs[y_hat_idx]

        # Get the active features
        x_acts_idxs = np.argwhere(x[0] > 0).flatten()
        x_acts_names = feature_names[x_acts_idxs]

        # Get the attributions of each predicted active feature
        A_hat = {}
        for x_act_name, x_act_value, y_coef_act in zip(x_acts_names, x[0, x_acts_idxs], y_hat_coef[x_acts_idxs]):
            A_hat[x_act_name] = (y_coef_act * x_act_value).item()

        # Get the top-k positive and negative attributions
        if top_k is not None:
            A_hat_pos = dict(sorted(A_hat.items(), key=lambda item: item[1], reverse=True)[:top_k])
            A_hat_neg = dict(sorted(A_hat.items(), key=lambda item: item[1], reverse=False)[:top_k])
            A_hat = {**A_hat_pos, **A_hat_neg}

        # Get the attributions of the true label
        if y is not None:
            # Get the index of the true class
            y_idx = self.class_list.tolist().index(y)

            # Get the coefficients for the true class
            y_coef = self.coefs[y_idx] # [n_features,]

            # Get the probability of the true label
            y_prob = y_hat_probs[y_idx]

            # Get the attributions of each supposed active feature
            A_true = {}
            for x_act_name, x_act_value, y_coef_act in zip(x_acts_names, x[0, x_acts_idxs], y_coef[x_acts_idxs]):
                A_true[x_act_name] = (y_coef_act * x_act_value).item()

            # Get the top-k positive and negative attributions
            if top_k is not None:
                A_true_pos = dict(sorted(A_true.items(), key=lambda item: item[1], reverse=True)[:top_k])
                A_true_neg = dict(sorted(A_true.items(), key=lambda item: item[1], reverse=False)[:top_k])
                A_true = {**A_true_pos, **A_true_neg}

        # Sort the attributions
        A_hat = dict(sorted(A_hat.items(), key=lambda item: item[1], reverse=True))
        A_true = dict(sorted(A_true.items(), key=lambda item: item[1], reverse=True))

        return {
            "t": t,
            "y_hat": y_hat,
            "y": y,
            "y_hat_prob": y_hat_prob.item(),
            "y_prob": y_prob.item(),
            "A_hat": A_hat,
            "A_true": A_true,
        }