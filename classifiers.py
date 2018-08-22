import sys
import os.path
import logging
from IPython.core.display import display, HTML
import numpy as np
import sklearn.metrics
from sklearn import svm, linear_model, tree, cluster
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from tabulate import tabulate


classifiers = {'Decision Tree': tree.DecisionTreeClassifier,
               'SVM': lambda : EvolutionaryAlgorithmSearchCV(estimator=svm.LinearSVC(),
                                           params={'C': np.logspace(-6, 6, num=10),
                                                   'loss': ['squared_hinge']},
                                           scoring='accuracy',
                                           verbose=1,
                                           population_size=50,
                                           n_jobs=10),
               'Logistic Regression': lambda : EvolutionaryAlgorithmSearchCV(
                   estimator=linear_model.LogisticRegression(),
                   params={'C': np.logspace(-6, 6, num=10)},
                   scoring='accuracy',
                   verbose=1,
                   population_size=50,
                   n_jobs=10)}
clusterers = {'K-Means': cluster.KMeans}


def is_interactive():
    import sys
    return 'ipykernel' in sys.modules


class ClassifierMetrics(object):
    def __init__(self, data, model):
        """

        Args:
            data (data.Data):
            model:
        """
        self.data = data
        self.x_train = self.data.x_train
        self.y_train = self.data.y_train
        self.cats_train = self.data.cats_nr_train
        self.x_test = self.data.x_test
        self.y_test = self.data.y_test
        self.cats_test = self.data.cats_nr_test

        self.model = model
        self.x_enc_train = self.model.get_embeddings(self.x_train)
        self.x_enc_test = self.model.get_embeddings(self.x_test)

        self.original_bin_scores = None
        self.original_cats_scores = None

        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def test_classifier(self, classifier, display_scores=True):
        out = {'metric': [], 'original': [], 'reduced': []}
        
        # original
        if self.original_bin_scores is None:
            clf = classifier()
            self.logger.info('Fitting %s to original data with bin labels' % clf)
            clf.fit(self.x_train, self.y_train)
            dt_preds_u = clf.predict(self.x_test)
            self.original_bin_scores = get_metrics(self.y_test, dt_preds_u)
        orig_scores = self.original_bin_scores

    
        # reduced
        clf_reduced = classifier()
        self.logger.info('Fitting %s to reduced data with bin labels' % clf_reduced)
        clf_reduced.fit(self.x_enc_train, self.y_train)
        dtr_preds_u = clf_reduced.predict(self.x_enc_test)
        reduced_scores = get_metrics(self.y_test, dtr_preds_u)
        
        for k in sorted(orig_scores):
            out['metric'].append(k)
            out['original'].append(orig_scores[k])
            out['reduced'].append(reduced_scores[k])

        out_cat = {'metric': [], 'original': [], 'reduced': []}
        # original with cats
        if self.original_cats_scores is None:
            clf_cats = classifier()
            self.logger.info('Fitting %s to original data with category labels' % clf_cats)
            clf_cats.fit(self.x_train, self.cats_train)
            dt_preds_u_cats = clf_cats.predict(self.x_test)
            self.original_cats_scores = get_metrics_cats(self.cats_test, dt_preds_u_cats)
        orig_scores_cats = self.original_cats_scores
        
        # reduced with cats
        clf_reduced_cats = classifier()
        self.logger.info('Fitting %s to reduced data with category labels' % clf_reduced_cats)
        clf_reduced_cats.fit(self.x_enc_train, self.cats_train)
        dtr_preds_u_cats = clf_reduced_cats.predict(self.x_enc_test)
        reduced_scores_cats = get_metrics_cats(self.cats_test, dtr_preds_u_cats)
        
        for k in sorted(orig_scores_cats):
            out_cat['metric'].append(k)
            out_cat['original'].append(orig_scores_cats[k])
            out_cat['reduced'].append(reduced_scores_cats[k])

        if display_scores:
            print('binary class:')
            if is_interactive():
                display(HTML(tabulate(out, headers='keys', tablefmt='html')))
            else:
                print(tabulate(out, headers='keys'))
            print('with attack categories:')
            if is_interactive():
                display(HTML(tabulate(out_cat, headers='keys', tablefmt='html')))
            else:
                print(tabulate(out_cat, headers='keys'))
        else:
            return out, out_cat


class ClustererMetrics(ClassifierMetrics):
    def test_classifier(self, classifier, display_scores=True):
        out = {'metric': [], 'original': [], 'reduced': []}
        out_cat = {'metric': [], 'original': [], 'reduced': []}  # never filled in

        # original
        clf = classifier()
        self.logger.info('Fitting %s to original data with bin labels' % clf)
        clf = clf.fit(self.x_train, self.y_train)
        dt_preds_u = clf.predict(self.x_test)
        orig_scores = get_clustering_metrics(self.x_test, self.y_test, dt_preds_u)

        # reduced
        clf_reduced = classifier()
        self.logger.info('Fitting %s to reduced data with bin labels' % clf_reduced)
        clf_reduced = clf_reduced.fit(self.x_enc_train, self.y_train)
        dtr_preds_u = clf_reduced.predict(self.x_enc_test)
        reduced_scores = get_clustering_metrics(self.x_enc_test, self.y_test, dtr_preds_u)

        for k in sorted(orig_scores):
            out['metric'].append(k)
            out['original'].append(orig_scores[k])
            out['reduced'].append(reduced_scores[k])


        if display_scores:
            print('binary class:')
            if is_interactive():
                display(HTML(tabulate(out, headers='keys', tablefmt='html')))
            else:
                print(tabulate(out, headers='keys'))
            print('with attack categories:')
            if is_interactive():
                display(HTML(tabulate(out_cat, headers='keys', tablefmt='html')))
            else:
                print(tabulate(out_cat, headers='keys'))
        else:
            return out, out_cat


class Aggregator(object):
    def __init__(self, model_class, number, *args, **kwargs):
        self.models = [model_class(*args, **kwargs) for _ in range(number)]
        self.metrics_classifiers = None
        self.scores = None
        self.histories = None

    def save_models(self, path):
        """
        Uses the ``save_model`` method of the models, and saves the multiple models into a directory.
        Inside the given directory (``path``), multiple files or directories will be creating, named with numbers from
        0 to the number of models in this aggregator.

        Args:
            path:

        Returns:

        """
        for i, model in enumerate(self.models):
            model.save_model(os.path.join(path, str(i)))

    def load_models(self, path):
        for i, model in enumerate(self.models):
            model.load_model(os.path.join(path, str(i)))

    def train(self, **kwargs):
        self.histories = [model.train(**kwargs) for model in self.models]

    def get_metrics(self, data):
        self.metrics_classifiers = [ClassifierMetrics(data, model) for model in self.models]
        self.metrics_clusterers = [ClustererMetrics(data, model) for model in self.models]
        self.scores = {}
        for cname, c in classifiers.items():
            self.scores[cname] = [met.test_classifier(c, display_scores=False) for met in self.metrics_classifiers]
        for cname, c in clusterers.items():
            self.scores[cname] = [met.test_classifier(c, display_scores=False) for met in self.metrics_clusterers]
        return self.scores

    def get_own_metrics(self, data, categories=True):
        """
        Used for the neural networks that include a label output.

        Args:
            data:
            categories: whether the predict function gives categories or binary labels (True when categories).

        Returns:

        """
        if categories:
            def aux_f(labels_scores):  # convert scores to just 0s and 1s
                labels = np.argmax(labels_scores, axis=1)
                new_labels = np.zeros(labels_scores.shape)
                new_labels[np.arange(len(labels)), labels] = 1
                return new_labels
            metrics = [get_metrics_cats(data.cats_test,
                                        aux_f(mod.predict_labels(data.x_test))) for mod in self.models]
        else:
            metrics = [get_metrics(data.y_test, np.rint(mod.predict_labels(data.x_test))) for mod in self.models]
        self.scores['own'] = []
        for met in metrics:
            new_table = {'metric': [], 'reduced': [], 'original': []}
            for k in sorted(met):  # iterate through keys (metric names)
                new_table['metric'].append(k)
                new_table['original'].append(-1)  # dummy value
                new_table['reduced'].append(met[k])
            if categories:
                self.scores['own'].append(({'metric': [], 'reduced': [], 'original': []}, new_table))
            else:
                self.scores['own'].append((new_table, {'metric': [], 'reduced': [], 'original': []}))

        return self.scores

    def apply_op(self, op, display_scores=True):
        out = {}
        for cname, metrics in self.scores.items():
            out[cname] = {}
            out[cname]['binary'] = {}
            out[cname]['binary']['reduced'] = {
                metrics[0][0]['metric'][met_idx]: op([met[0]['reduced'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][0]['reduced']))
            }
            out[cname]['binary']['original'] = {
                metrics[0][0]['metric'][met_idx]: op([met[0]['original'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][0]['original']))
            }

            out[cname]['multi'] = {}
            out[cname]['multi']['reduced'] = {
                metrics[0][1]['metric'][met_idx]: op([met[1]['reduced'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][1]['reduced']))
            }
            out[cname]['multi']['original'] = {
                metrics[0][1]['metric'][met_idx]: op([met[1]['original'][met_idx] for met in metrics]) \
                for met_idx in range(len(metrics[0][1]['original']))
            }
        if not display_scores:
            return out
        else:
            for name in out:
                print('aggregated metrics for %s' % name)
                tmp = out[name]

                b = tmp['binary']
                b_out = {'metric': [], 'original': [], 'reduced': []}
                for k in sorted(b['reduced']):
                    b_out['metric'].append(k)
                    b_out['original'].append(b['original'][k])
                    b_out['reduced'].append(b['reduced'][k])

                m = tmp['multi']
                m_out = {'metric': [], 'original': [], 'reduced': []}
                for k in sorted(m['reduced']):
                    m_out['metric'].append(k)
                    m_out['original'].append(m['original'][k])
                    m_out['reduced'].append(m['reduced'][k])

                print('binary class:')
                if is_interactive():
                    display(HTML(tabulate(b_out, headers='keys', tablefmt='html')))
                else:
                    print(tabulate(b_out, headers='keys'))
                print('with attack categories:')
                if is_interactive():
                    display(HTML(tabulate(m_out, headers='keys', tablefmt='html')))
                else:
                    print(tabulate(m_out, headers='keys'))


    def mean(self, display_scores=True):
        return self.apply_op(np.mean, display_scores=display_scores)

    def std(self, display_scores=True):
        return self.apply_op(np.std, display_scores=display_scores)


def test_model(data, model):
    for name, c in classifiers.items():
        metrics = ClassifierMetrics(data, model)
        print('metrics for %s' % name)
        metrics.test_classifier(c)


def get_metrics(y_true, y_pred):
    return {
        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred),
        'precision': sklearn.metrics.precision_score(y_true, y_pred),
        'recall': sklearn.metrics.recall_score(y_true, y_pred),
        'f1': sklearn.metrics.f1_score(y_true, y_pred)
    }


def get_clustering_metrics(X, y_true, y_pred):
    return {
        'adj_rand_index': sklearn.metrics.adjusted_rand_score(y_true, y_pred),
        'adj_mutual_info': sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred),
        'homogeneity': sklearn.metrics.homogeneity_score(y_true, y_pred),
        'completeness': sklearn.metrics.completeness_score(y_true, y_pred),
        'v-measure': sklearn.metrics.v_measure_score(y_true, y_pred),
        'fowlkes-mallows': sklearn.metrics.fowlkes_mallows_score(y_true, y_pred),
        'silhouette': sklearn.metrics.silhouette_score(X, y_pred, sample_size=20000),
        'calinski-harabaz': sklearn.metrics.calinski_harabaz_score(X, y_pred)
    }


def get_metrics_cats(y_true, y_pred):
    out = {'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred)}
    out.update({'f1_%d' % i : v for i, v in enumerate(
        sklearn.metrics.f1_score(y_true, y_pred, average=None)
    )})
    out.update({'precision_%d' % i : v for i, v in enumerate(
        sklearn.metrics.precision_score(y_true, y_pred, average=None)
    )})
    out.update({'recall_%d' % i : v for i, v in enumerate(
        sklearn.metrics.recall_score(y_true, y_pred, average=None)
    )})
    out['f1_macro'] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    out['f1_micro'] = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    out['precision_macro'] = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    out['precision_micro'] = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
    out['recall_macro'] = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    out['recall_micro'] = sklearn.metrics.recall_score(y_true, y_pred, average='micro')
    return out
