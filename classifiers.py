import sys
import os.path
import logging
from collections import OrderedDict
from IPython.core.display import display, HTML
import numpy as np
import sklearn.metrics
from scipy.spatial import cKDTree
from sklearn import svm, linear_model, tree, cluster, model_selection
from tabulate import tabulate


classifiers = OrderedDict([
    ('Decision Tree', lambda : model_selection.GridSearchCV(
        estimator=tree.DecisionTreeClassifier(),
        param_grid={
            'max_depth': [int(np.ceil(x)) for x in np.logspace(1, 8, num=20, base=2)],
            'min_samples_leaf': [int(np.ceil(x)) for x in np.logspace(1, 6, num=15, base=2)]
        }
    )),
    # ('SVM', lambda : model_selection.GridSearchCV(estimator=svm.LinearSVC(),
    #                                               param_grid={'C': np.logspace(-6, 6, num=20, base=10),
    #                                                           'loss': ['squared_hinge'],
    #                                                           'random_state': [0]},
    #                                               scoring='accuracy',
    #                                               verbose=3,
    #                                               n_jobs=10)),
    # ('Logistic Regression', lambda : model_selection.GridSearchCV(
    #     estimator=linear_model.LogisticRegression(),
    #     param_grid={'C': np.logspace(-6, 6, num=20, base=10)},
    #     scoring='accuracy',
    #     verbose=3,
    #     n_jobs=10))
])
clusterers = OrderedDict([
    # ('K-Means', cluster.KMeans)
])


def is_interactive():
    import sys
    return 'ipykernel' in sys.modules


class ClassifierMetrics(object):
    def __init__(self, data, model, fixed_scores):
        """

        Args:
            data (data.Data):
            model:
        """
        self.data = data
        self.x_train = self.data.x_train
        self.y_train = self.data.y_train
        self.cats_train = self.data.cats_nr_train
        self.x_val = self.data.x_val
        self.y_val = self.data.y_val
        self.cats_val = self.data.cats_nr_val
        self.x_test = self.data.x_test
        self.y_test = self.data.y_test
        self.cats_test = self.data.cats_nr_test

        self.model = model
        self.x_enc_train = self.model.get_embeddings(self.x_train)
        self.x_enc_val = self.model.get_embeddings(self.x_val)
        self.x_enc_test = self.model.get_embeddings(self.x_test)

        self.fixed_scores = fixed_scores

        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def _after_fit(self, clf):
        if 'GridSearchCV' in str(clf):
            self.logger.warning('best parameters for %s are: %s' % (
                str(clf),
                str(clf.best_params_)
            ))

    def test_classifier(self, classifier, display_scores=True):
        out = OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                           ('reduced', []), ('reduced_val', []), ('reduced_train', [])])

        # original
        if 'bin_original' not in self.fixed_scores:
            self.fixed_scores['bin_original'] = OrderedDict()
        if str(classifier) not in self.fixed_scores['bin_original']:
            clf = classifier()
            self.logger.info('Fitting %s to original data with bin labels' % clf)
            clf.fit(self.x_train, self.y_train)
            self._after_fit(clf)
            dt_preds_u = clf.predict(self.x_test)
            self.fixed_scores['bin_original'][str(classifier)] = get_metrics(self.y_test, dt_preds_u)
            dt_preds_u_val = clf.predict(self.x_val)
            self.fixed_scores['bin_original'][str(classifier) + '_val'] = get_metrics(self.y_val, dt_preds_u_val)
            dt_preds_u_train = clf.predict(self.x_train)
            self.fixed_scores['bin_original'][str(classifier) + '_train'] = get_metrics(self.y_train, dt_preds_u_train)
        orig_scores = self.fixed_scores['bin_original'][str(classifier)]
        orig_scores_val = self.fixed_scores['bin_original'][str(classifier) + '_val']
        orig_scores_train = self.fixed_scores['bin_original'][str(classifier) + '_train']

        # reduced
        clf_reduced = classifier()
        self.logger.info('Fitting %s to reduced data with bin labels' % clf_reduced)
        clf_reduced.fit(self.x_enc_train, self.y_train)
        self._after_fit(clf_reduced)
        dtr_preds_u = clf_reduced.predict(self.x_enc_test)
        reduced_scores = get_metrics(self.y_test, dtr_preds_u)
        dtr_preds_u_val = clf_reduced.predict(self.x_enc_val)
        reduced_scores_val = get_metrics(self.y_val, dtr_preds_u_val)
        dtr_preds_u_train = clf_reduced.predict(self.x_enc_train)
        reduced_scores_train = get_metrics(self.y_train, dtr_preds_u_train)

        for k in sorted(orig_scores):
            out['metric'].append(k)
            out['original'].append(orig_scores[k])
            out['original_val'].append(orig_scores_val[k])
            out['original_train'].append(orig_scores_train[k])
            out['reduced'].append(reduced_scores[k])
            out['reduced_val'].append(reduced_scores_val[k])
            out['reduced_train'].append(reduced_scores_train[k])

        out_cat = OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                               ('reduced', []), ('reduced_val', []), ('reduced_train', [])])
        # original with cats
        if 'cats_original' not in self.fixed_scores:
            self.fixed_scores['cats_original'] = OrderedDict()
        if str(classifier) not in self.fixed_scores['cats_original']:
            clf_cats = classifier()
            self.logger.info('Fitting %s to original data with category labels' % clf_cats)
            clf_cats.fit(self.x_train, self.cats_train)
            self._after_fit(clf_cats)
            dt_preds_u_cats = clf_cats.predict(self.x_test)
            self.fixed_scores['cats_original'][str(classifier)] = get_metrics_cats(self.cats_test, dt_preds_u_cats)
            dt_preds_u_cats_val = clf_cats.predict(self.x_val)
            self.fixed_scores['cats_original'][str(classifier) + '_val'] = \
                get_metrics_cats(self.cats_val, dt_preds_u_cats_val)
            dt_preds_u_cats_train = clf_cats.predict(self.x_train)
            self.fixed_scores['cats_original'][str(classifier) + '_train'] = \
                get_metrics_cats(self.cats_train, dt_preds_u_cats_train)
        orig_scores_cats = self.fixed_scores['cats_original'][str(classifier)]
        orig_scores_cats_val = self.fixed_scores['cats_original'][str(classifier) + '_val']
        orig_scores_cats_train = self.fixed_scores['cats_original'][str(classifier) + '_train']

        # reduced with cats
        clf_reduced_cats = classifier()
        self.logger.info('Fitting %s to reduced data with category labels' % clf_reduced_cats)
        clf_reduced_cats.fit(self.x_enc_train, self.cats_train)
        self._after_fit(clf_reduced_cats)
        dtr_preds_u_cats = clf_reduced_cats.predict(self.x_enc_test)
        reduced_scores_cats = get_metrics_cats(self.cats_test, dtr_preds_u_cats)
        dtr_preds_u_cats_val = clf_reduced_cats.predict(self.x_enc_val)
        reduced_scores_cats_val = get_metrics_cats(self.cats_val, dtr_preds_u_cats_val)
        dtr_preds_u_cats_train = clf_reduced_cats.predict(self.x_enc_train)
        reduced_scores_cats_train = get_metrics_cats(self.cats_train, dtr_preds_u_cats_train)

        for k in sorted(orig_scores_cats):
            out_cat['metric'].append(k)
            out_cat['original'].append(orig_scores_cats[k])
            out_cat['original_val'].append(orig_scores_cats_val[k])
            out_cat['original_train'].append(orig_scores_cats_train[k])
            out_cat['reduced'].append(reduced_scores_cats[k])
            out_cat['reduced_val'].append(reduced_scores_cats_val[k])
            out_cat['reduced_train'].append(reduced_scores_cats_train[k])

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
        out = OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                           ('reduced', []), ('reduced_val', []), ('reduced_train', [])])
        out_cat = OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                               ('reduced', []), ('reduced_val', []), ('reduced_train', [])])  # never filled in

        # original
        if 'clust_bin_original' not in self.fixed_scores:
            self.fixed_scores['clust_bin_original'] = OrderedDict()
        if str(classifier) not in self.fixed_scores['clust_bin_original']:
            clf = classifier()
            self.logger.info('Fitting %s to original data with bin labels' % clf)
            clf = clf.fit(self.x_train, self.y_train)
            self._after_fit(clf)
            dt_preds_u = clf.predict(self.x_test)
            self.fixed_scores['clust_bin_original'][str(classifier)] = \
                get_clustering_metrics(self.x_test, self.y_test, dt_preds_u)
            dt_preds_u_val = clf.predict(self.x_val)
            self.fixed_scores['clust_bin_original'][str(classifier) + '_val'] = \
                get_clustering_metrics(self.x_val, self.y_val, dt_preds_u_val)
            dt_preds_u_train = clf.predict(self.x_train)
            self.fixed_scores['clust_bin_original'][str(classifier) + '_train'] = \
                get_clustering_metrics(self.x_train, self.y_train, dt_preds_u_train)
        orig_scores = self.fixed_scores['clust_bin_original'][str(classifier)]
        orig_scores_val = self.fixed_scores['clust_bin_original'][str(classifier) + '_val']
        orig_scores_train = self.fixed_scores['clust_bin_original'][str(classifier) + '_train']

        # reduced
        clf_reduced = classifier()
        self.logger.info('Fitting %s to reduced data with bin labels' % clf_reduced)
        clf_reduced = clf_reduced.fit(self.x_enc_train, self.y_train)
        self._after_fit(clf_reduced)
        dtr_preds_u = clf_reduced.predict(self.x_enc_test)
        reduced_scores = get_clustering_metrics(self.x_enc_test, self.y_test, dtr_preds_u)
        dtr_preds_u_val = clf_reduced.predict(self.x_enc_val)
        reduced_scores_val = get_clustering_metrics(self.x_enc_val, self.y_val, dtr_preds_u_val)
        dtr_preds_u_train = clf_reduced.predict(self.x_enc_train)
        reduced_scores_train = get_clustering_metrics(self.x_enc_train, self.y_train, dtr_preds_u_train)

        for k in sorted(orig_scores):
            out['metric'].append(k)
            out['original'].append(orig_scores[k])
            out['original_val'].append(orig_scores_val[k])
            out['original_train'].append(orig_scores_train[k])
            out['reduced'].append(reduced_scores[k])
            out['reduced_val'].append(reduced_scores_val[k])
            out['reduced_train'].append(reduced_scores_train[k])


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

        self.fixed_scores = OrderedDict()  # used for keeping track of scores that are the same across the multiple
        # models (no need to recompute)

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
        self.metrics_classifiers = [ClassifierMetrics(data, model, self.fixed_scores) for model in self.models]
        self.metrics_clusterers = [ClustererMetrics(data, model, self.fixed_scores) for model in self.models]
        self.scores = OrderedDict()
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
            metrics_val = [get_metrics_cats(data.cats_val,
                                        aux_f(mod.predict_labels(data.x_val))) for mod in self.models]
            metrics_train = [get_metrics_cats(data.cats_train,
                                        aux_f(mod.predict_labels(data.x_train))) for mod in self.models]
        else:
            metrics = [get_metrics(data.y_test, np.rint(mod.predict_labels(data.x_test))) for mod in self.models]
            metrics_val = [get_metrics(data.y_val, np.rint(mod.predict_labels(data.x_val))) for mod in self.models]
            metrics_train = [get_metrics(data.y_train, np.rint(mod.predict_labels(data.x_train))) for mod in self.models]
        self.scores['own'] = []
        for met, met_val, met_train in zip(metrics, metrics_val, metrics_train):
            new_table = OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                                     ('reduced', []), ('reduced_val', []), ('reduced_train', [])])
            for k in sorted(met):  # iterate through keys (metric names)
                new_table['metric'].append(k)
                new_table['original'].append(-1)  # dummy value
                new_table['original_val'].append(-1)
                new_table['original_train'].append(-1)
                new_table['reduced'].append(met[k])
                new_table['reduced_val'].append(met_val[k])
                new_table['reduced_train'].append(met_train[k])
            if categories:
                self.scores['own'].append((OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                                                        ('reduced', []), ('reduced_val', []), ('reduced_train', [])]),
                                           new_table))
            else:
                self.scores['own'].append((new_table,
                                           OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                                                        ('reduced', []), ('reduced_val', []), ('reduced_train', [])])))

        return self.scores

    def apply_op(self, op, display_scores=True):
        out = OrderedDict()
        for cname, metrics in self.scores.items():
            out[cname] = OrderedDict()
            out[cname]['binary'] = OrderedDict()
            out[cname]['binary']['reduced'] = OrderedDict([
                (metrics[0][0]['metric'][met_idx], op([met[0]['reduced'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][0]['reduced']))
            ])
            out[cname]['binary']['reduced_val'] = OrderedDict([
                (metrics[0][0]['metric'][met_idx], op([met[0]['reduced_val'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][0]['reduced_val']))
            ])
            out[cname]['binary']['reduced_train'] = OrderedDict([
                (metrics[0][0]['metric'][met_idx], op([met[0]['reduced_train'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][0]['reduced_train']))
            ])
            out[cname]['binary']['original'] = OrderedDict([
                (metrics[0][0]['metric'][met_idx], op([met[0]['original'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][0]['original']))
            ])
            out[cname]['binary']['original_val'] = OrderedDict([
                (metrics[0][0]['metric'][met_idx], op([met[0]['original_val'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][0]['original_val']))
            ])
            out[cname]['binary']['original_train'] = OrderedDict([
                (metrics[0][0]['metric'][met_idx], op([met[0]['original_train'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][0]['original_train']))
            ])

            out[cname]['multi'] = OrderedDict()
            out[cname]['multi']['reduced'] = OrderedDict([
                (metrics[0][1]['metric'][met_idx], op([met[1]['reduced'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][1]['reduced']))
            ])
            out[cname]['multi']['reduced_val'] = OrderedDict([
                (metrics[0][1]['metric'][met_idx], op([met[1]['reduced_val'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][1]['reduced_val']))
            ])
            out[cname]['multi']['reduced_train'] = OrderedDict([
                (metrics[0][1]['metric'][met_idx], op([met[1]['reduced_train'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][1]['reduced_train']))
            ])
            out[cname]['multi']['original'] = OrderedDict([
                (metrics[0][1]['metric'][met_idx], op([met[1]['original'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][1]['original']))
            ])
            out[cname]['multi']['original_val'] = OrderedDict([
                (metrics[0][1]['metric'][met_idx], op([met[1]['original_val'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][1]['original_val']))
            ])
            out[cname]['multi']['original_train'] = OrderedDict([
                (metrics[0][1]['metric'][met_idx], op([met[1]['original_train'][met_idx] for met in metrics])) \
                for met_idx in range(len(metrics[0][1]['original_train']))
            ])
        if not display_scores:
            return out
        else:
            for name in out:
                print('aggregated metrics for %s' % name)
                tmp = out[name]

                b = tmp['binary']
                b_out = OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                                     ('reduced', []), ('reduced_val', []), ('reduced_train', [])])
                for k in sorted(b['reduced']):
                    b_out['metric'].append(k)
                    b_out['original'].append(b['original'][k])
                    b_out['original_val'].append(b['original_val'][k])
                    b_out['original_train'].append(b['original_train'][k])
                    b_out['reduced'].append(b['reduced'][k])
                    b_out['reduced_val'].append(b['reduced_val'][k])
                    b_out['reduced_train'].append(b['reduced_train'][k])

                m = tmp['multi']
                m_out = OrderedDict([('metric', []), ('original', []), ('original_val', []), ('original_train', []),
                                     ('reduced', []), ('reduced_val', []), ('reduced_train', [])])
                for k in sorted(m['reduced']):
                    m_out['metric'].append(k)
                    m_out['original'].append(m['original'][k])
                    m_out['original_val'].append(m['original_val'][k])
                    m_out['original_train'].append(m['original_train'][k])
                    m_out['reduced'].append(m['reduced'][k])
                    m_out['reduced_val'].append(m['reduced_val'][k])
                    m_out['reduced_train'].append(m['reduced_train'][k])

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
        metrics = ClassifierMetrics(data, model, OrderedDict())
        print('metrics for %s' % name)
        metrics.test_classifier(c)


def get_metrics(y_true, y_pred):
    def far(y_true, y_pred):
        conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_mat.ravel()
        return ((fp / (fp + tn)) + (fn / (fn + tp))) / 2
    return OrderedDict([
        ('accuracy', sklearn.metrics.accuracy_score(y_true, y_pred)),
        ('precision', sklearn.metrics.precision_score(y_true, y_pred)),
        ('recall', sklearn.metrics.recall_score(y_true, y_pred)),
        ('f1', sklearn.metrics.f1_score(y_true, y_pred)),
        ('far', far(y_true, y_pred)),
    ])


def get_clustering_metrics(X, y_true, y_pred):
    return OrderedDict([
        ('adj_rand_index', sklearn.metrics.adjusted_rand_score(y_true, y_pred)),
        ('adj_mutual_info', sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)),
        ('homogeneity', sklearn.metrics.homogeneity_score(y_true, y_pred)),
        ('completeness', sklearn.metrics.completeness_score(y_true, y_pred)),
        ('v-measure', sklearn.metrics.v_measure_score(y_true, y_pred)),
        ('fowlkes-mallows', sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)),
        ('silhouette', sklearn.metrics.silhouette_score(X, y_pred, sample_size=20000)),
        ('calinski-harabaz', sklearn.metrics.calinski_harabaz_score(X, y_pred))
    ])


def get_metrics_cats(y_true, y_pred):
    out = OrderedDict([('accuracy', sklearn.metrics.accuracy_score(y_true, y_pred))])
    out.update(OrderedDict([('f1_%d' % i , v) for i, v in enumerate(
        sklearn.metrics.f1_score(y_true, y_pred, average=None))])
    )
    out.update(OrderedDict([('precision_%d' % i , v) for i, v in enumerate(
        sklearn.metrics.precision_score(y_true, y_pred, average=None))])
    )
    out.update(OrderedDict([('recall_%d' % i , v) for i, v in enumerate(
        sklearn.metrics.recall_score(y_true, y_pred, average=None))])
    )
    out['f1_macro'] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    out['f1_micro'] = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    out['precision_macro'] = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    out['precision_micro'] = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
    out['recall_macro'] = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    out['recall_micro'] = sklearn.metrics.recall_score(y_true, y_pred, average='micro')
    return out


class VisualClassifier(object):
    """Takes 2D train data, and for each new test point returns probability of belonging to each class.
    Probabilities are calculated by taking the classes from the samples in the train data within eps (given as input)
    distance of the test point."""
    def __init__(self, leafsize=16):
        """

        Args:
            leafsize (int): leafsize parameter of scipy.spatial.KDTree
        """
        self.leafsize = leafsize
        self.tree = None
        self.labels = None
        self.possible_labels = None

    def fit(self, data, labels):
        if data.shape[1] != 2:
            raise ValueError('Given training data must be 2-dimensional!')
        if self.tree is not None:
            raise AssertionError('Train has already been called once!')
        self.labels = np.array(labels, dtype=int)
        self.possible_labels = np.unique(self.labels)
        self.tree = cKDTree(data, self.leafsize)

    def predict_proba(self, x, eps=0.05, return_counts=False):
        if len(x.shape) == 2:
            output = []
            for xx in x:
                indices = self.tree.query_ball_point(xx, r=eps)
                if not return_counts:
                    output.append(self._predict_proba_single(indices))
                else:
                    output.append((self._predict_proba_single(indices), len(indices)))
            return output
        elif len(x.shape) == 1:
            indices = self.tree.query_ball_point(x, r=eps)
            if not return_counts:
                return self._predict_proba_single(indices)
            else:
                return self._predict_proba_single(indices), len(indices)
        else:
            raise ValueError('Input has invalid dimensions!')

    def predict(self, x, eps=0.05):
        probabilities = self.predict_proba(x, eps)

        def get_class(probs):
            if probs.sum() == 0:
                return -1
            else:
                return np.argmax(probs)

        if len(x.shape) == 1:
            return get_class(probabilities)
        elif len(x.shape) == 2:
            return np.apply_along_axis(get_class, 1, probabilities)
        else:
            raise ValueError('Input has invalid dimensions!')

    def _predict_proba_single(self, indices):
        values, counts = np.unique(np.array([self.labels[i] for i in indices]), return_counts=True)
        counts = counts / len(indices)  # normalize to probability
        arr = np.zeros(len(self.possible_labels))
        for v, c in zip(values, counts):
            arr[v] += c
        return arr
