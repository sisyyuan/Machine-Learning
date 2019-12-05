#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional
from abc import ABCMeta, abstractmethod
import logging
import json
from threading import Lock
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_predict
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.tree
import sklearn.neighbors

from collections.abc import Iterable
from early_stopping import EarlyStopping
from itertools import tee
from sklearn.metrics.classification import classification_report, f1_score
from functools import reduce
import operator
from collections import namedtuple
from multiprocessing import Pool
import pandas as pd
import random
from itertools import combinations
from tqdm import tqdm
from itertools import islice
from dask import dataframe as dd
from multiprocessing import cpu_count
from dask.diagnostics import ProgressBar


ClassifierSettings = namedtuple('ClassifierSettings', ['clf', 'module_kwargs', 'fit_kwargs', 'enable'])


class DataGenerator:

    @staticmethod
    def _read_obj(file_path):
        with open(file_path, 'r') as f:
            num = int(f.readline())
            trans = {}
            for _ in range(num):
                name, id_ = f.readline().split()
                name = name.strip()
                id_ = int(id_)
                trans[name] = id_
                trans[id_] = name
            return trans


    def _read_data(self,
                   dataset_df=None,
                   vectors_file=None,
                   entities_file=None,
                   negative_data_ratio=0,
                   ):
        self.data_lock.acquire()

        if dataset_df is not None:
            self.links = dataset_df.apply(lambda x: x.str.strip()).drop_duplicates()
            self.links['target'] = pd.Series(data=(1,)*self.links.shape[0])

        if vectors_file is not None:
            with open(vectors_file, 'r') as f:
                data = json.load(f)
                self.ent_vec = np.array(data['ent_embeddings'])
        if entities_file is not None:
            self.ent_dict = self._read_obj(entities_file)

        if negative_data_ratio > 0:
            logging.info('Generating negative samples...')
            curr = set(map(tuple, self.links.to_numpy()))
            neg_data_num = len(self.links) * negative_data_ratio
            data = tuple(filter(lambda x: isinstance(x, str), self.ent_dict.keys()))
            results = []
            while neg_data_num:
                t = tuple(random.sample(data, k=2))
                if t not in curr:
                    results.append(t + (0,))
                    neg_data_num -= 1
            df = pd.DataFrame(results, columns=self.links.columns)
            self.links = pd.concat([self.links, df])
            logging.info('Generating negative samples finished.')


        self.data_lock.release()

    def __init__(self,
                 dataset_df=None,
                 vectors_file=None,
                 entities_file=None,
                 ):
        self.ent_vec = None
        self.ent_dict = {}  # dict key contains only two type: str, int; so, merged
        self.links = None
        self.data_lock = Lock()

        self._read_data(dataset_df=dataset_df,
                        vectors_file=vectors_file,
                        entities_file=entities_file,
                        negative_data_ratio=2,
                        )

        if any(map(lambda x: x is None or len(x) == 0,
                       (self.ent_vec, self.ent_dict, self.links))):
            raise RuntimeError('No enough source specified')

        self.splitter = RepeatedKFold(n_splits=10, n_repeats=1)
        self.iter = self.splitter.split(self.links)


    def __iter__(self):
        return self

    def __next__(self):
        """Return K folds encoded data"""
        train_index, test_index = next(self.iter)
        return (iter(self.links.to_numpy()[train_index, :-1]), iter(self.links.to_numpy()[train_index, -1])),\
               (iter(self.links.to_numpy()[test_index, :-1]), iter(self.links.to_numpy()[test_index,  -1]))


    def ent_cnt(self):
        return len(self.ent_vec)

    def link_cnt(self):
        return len(self.links)

    def translate(self, objs):
        if isinstance(objs, str) or not isinstance(objs, Iterable):
            objs = (objs,)
        res = []
        for obj in objs:
            res.append(self.ent_dict[obj])
        return res


    def embed(self, features, target=None):
        if not isinstance(features, Iterable):
            raise RuntimeError('features should be iterable')

        if target is not None:
            target = iter(target)

        for feature in features:
            if target is not None:
                yield (np.array([self.ent_vec[self.translate(x)] for x in feature]), next(target))
            else:
                yield np.array([self.ent_vec[self.translate(x)] for x in feature])

    def get_all(self):
        gen = self.embed(self.links.to_numpy()[:, :-1], self.links.to_numpy()[:, -1])
        xs = []
        ys = []
        for x, y in gen:
            xs.append(x.reshape(reduce(operator.mul, x.shape)))
            ys.append(y)
        xs = np.stack(xs)
        ys = np.stack(ys)
        return xs, ys




class AbstractClassifier(metaclass=ABCMeta):
    def __init__(self):
        logging.basicConfig(format="%(asctime)s %(levelname)8s: %(message)s", level=logging.DEBUG)

    @abstractmethod
    def fit(self, input_xy, **kwargs):
        pass

    @abstractmethod
    def predict(self, input_x, **kwargs):
        pass

    @abstractmethod
    def test(self, input_xy, **kwargs):
        pass



class NNClassifier(AbstractClassifier):

    class Net(nn.Module):
        def __init__(self,
                     in_vectors=2,
                     in_vector_size=500,
                     compressed_size=300,
                     hidden_layers_size=(256, 128),
                     out_vector_size=16,
                     device=None,
                     ):
            super(self.__class__, self).__init__()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
            self.compress_layers = [nn.Linear(in_features=in_vector_size, out_features=compressed_size)
                                    for _ in range(in_vectors)]
            self.hidden_layers = []
            in_size = out_size = compressed_size * in_vectors
            for out_size in hidden_layers_size:
                self.hidden_layers.append(nn.Linear(in_features=in_size, out_features=out_size))
                in_size = out_size
            self.output_layer = nn.Linear(in_features=out_size, out_features=out_vector_size)


        def forward(self, input_):
            compressed = []
            for i, x in enumerate(input_):
                compressed.append(self.compress_layers[i](torch.from_numpy(x).view(1, 1, -1).float()
                                                          .to(device=self.device)))
            hidden = torch.cat(compressed, dim=-1)
            for i, layer in enumerate(self.hidden_layers):
                hidden = layer(hidden)

            output = self.output_layer(hidden)
            return output



    def __init__(self):
        super(self.__class__, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_module = self.Net(device=self.device)


    def fit(self, input_xy,
            lr=0.01,
            weight=None,
            epoch='auto',
            print_batch_num=10,
            batch_size=128,
            tot_size=np.inf,
            max_epoch=20,
            test_input=None,
            target_names=None,
            enable_early_stopping=False,
            **kwargs):
        optimizer = torch.optim.SGD(self.nn_module.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight,
                                                            dtype=torch.float32,
                                                            device=self.device
                                                            ) if weight is not None else None)

        if test_input is not None:
            # use test returned f1 score if validation set specified
            early_stopping = EarlyStopping(mode='max', patience=5, percentage=True)
            test_input = tee(test_input, max_epoch if test_input is not None else 0)
        else:
            # else use average loss
            early_stopping = EarlyStopping(mode='min', patience=5, percentage=False)


        logging.info('Train starting...')

        for epoch_idx, epoch_input in enumerate(tee(input_xy, max_epoch)):
            tot_loss = []
            for batch_cnt, batch_data in enumerate(zip(*([iter(epoch_input)] * batch_size))):
                outputs = []
                trues = []

                optimizer.zero_grad()
                for x, y in batch_data:
                    output = self.nn_module(x).view(-1)
                    outputs.append(output)
                    trues.append(y.argmax())
                outputs = torch.stack(outputs)
                trues = torch.tensor(trues, dtype=torch.long, device=self.device)
                loss = criterion(outputs, trues)

                if (batch_cnt+1) % print_batch_num == 0:
                    logging.debug('Epoch %5.3f, Current loss: %10.8f' %
                                  (epoch_idx+batch_cnt*batch_size/tot_size, loss))


                tot_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            if test_input is not None:
                score = self.test(test_input[epoch_idx], target_names=target_names)
                logging.info('Epoch %d, test score %4.2f' % (epoch_idx, score))
            else:
                score = sum(tot_loss) / len(tot_loss)
                logging.info('Epoch %d, avg loss %4.2f' % (epoch_idx, score))

            if enable_early_stopping and early_stopping.step(score):
                logging.info('Early stopped.')
                break

        logging.info('Train end.')

    def predict(self, input_x, prob=False, **kwargs):
        with torch.no_grad():
            results = []
            for x in input_x:
                result = nn.functional.softmax(self.nn_module(x).view(-1), dim=0)
                results.append(result)
            results = torch.stack(results).data.numpy()
            return results if prob else results.argmax(axis=1)

    def test(self, input_xy, target_names=None, **kwargs):
        input_x = []
        trues = []
        for x, y in input_xy:
            input_x.append(x)
            trues.append(y.argmax())
        results = self.predict(input_x, prob=False)
        trues = np.array(trues)
        print(classification_report(trues, results, target_names=target_names))
        return f1_score(trues, results, average="micro")


class SciKitAbstractClassifier(AbstractClassifier):

    def __init__(self, **kwargs):
        super(SciKitAbstractClassifier, self).__init__()
        self.module = None


    def fit(self, input_, *args, **kwargs):
        if len(args) == 1:
            xs = input_
            ys = args[0]
        elif len(args) == 0:
            xs, ys = zip(*input_)
        else:
            raise RuntimeError('Too many args')
        xs = np.stack(xs).reshape(-1, reduce(operator.mul, xs[0].shape))
        ys = np.stack(ys)
        xs_inv = np.concatenate([xs[:, xs.shape[1] // 2:], xs[:, :xs.shape[1] // 2]], axis=1)
        xs = np.concatenate([xs, xs_inv], axis=0)
        ys = np.concatenate([ys, ys], axis=0)
        self.module.fit(xs, ys)

    def predict(self, input_x, **kwargs):
        return self.module.predict(input_x)

    def predict_prob(self, input_x, **kwargs):
        return self.module.predict_proba(input_x)

    def test(self, input_xy, target_names=None, **kwargs):
        xs, trues = zip(*input_xy)
        xs = np.stack(xs).reshape(-1, reduce(operator.mul, xs[0].shape))
        trues = np.stack(trues)
        results = self.predict(xs)
        print(classification_report(trues, results, target_names=target_names))
        return f1_score(trues, results, average="micro")

    def get_params(self, *args, **kwargs):
        return self.module.get_params(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        self.module.set_params(*args, **kwargs)
        return self


class SVMClassifier(SciKitAbstractClassifier):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        self.module = sklearn.svm.LinearSVC()
        self.set_params(**kwargs)


class RandomForestClassifier(SciKitAbstractClassifier):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        self.module = sklearn.ensemble.RandomForestClassifier()
        self.set_params(**kwargs)


class GBDTClassifier(SciKitAbstractClassifier):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        self.module = sklearn.ensemble.GradientBoostingClassifier()
        self.set_params(**kwargs)


class GaussianNB(SciKitAbstractClassifier):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        self.module = sklearn.naive_bayes.GaussianNB()
        self.set_params(**kwargs)


class DecisionTreeClassifier(SciKitAbstractClassifier):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        self.module = sklearn.tree.DecisionTreeClassifier()
        self.set_params(**kwargs)


class KNeighborsClassifier(SciKitAbstractClassifier):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()
        self.module = sklearn.neighbors.KNeighborsClassifier()
        self.set_params(**kwargs)


class ClassifierTester:
    def __init__(self, dg):
        self.dg = dg

    def test_module(self, module, module_kwargs, fit_kwargs):
        train, test = next(self.dg)
        logging.info('Testing module %s...' % module.__name__)
        module_object = module(**module_kwargs)
        module_object.fit(
            self.dg.embed(*train),
            **fit_kwargs
        )
        module_object.test(
            self.dg.embed(*test),
            target_names=None,
        )
        logging.info('Testing module %s finished.' % module.__name__)

    def cv_test_module(self, module, module_kwargs, fit_kwargs):
        logging.info('Cross-validate testing module %s...' % module.__name__)

        xs, ys = self.dg.get_all()

        module_object = module(**module_kwargs)
        predict = cross_val_predict(
            module_object,
            xs, ys,
            cv=10,
            n_jobs=-1,
        )

        print(classification_report(ys, predict))
        logging.info('Cross-validate testing module %s finished' % module.__name__)

def run_test():
    logging.basicConfig(format="%(asctime)s %(levelname)8s: %(message)s", level=logging.DEBUG)

    dg = DataGenerator(
        vectors_file='/mnt/d/Downloads/embedding.vec.json',
        #vectors_file='embedding.vec.json',
        entities_file='drugs.txt',
        dataset_df=pd.read_excel('41467_2019_9186_MOESM5_ESM.xlsx', skipfooter=1),
    )

    classifiers = [
        # note that some of them take a long time
        ClassifierSettings(
            clf=NNClassifier,
            module_kwargs={},
            fit_kwargs={
                'tot_size': dg.link_cnt() * 0.9,
                'batch_size': 1,
                'print_batch_num': 10240,
            },
            enable=False,
        ),
        ClassifierSettings(
            clf=SVMClassifier,
            module_kwargs={},
            fit_kwargs={},
            enable=True,
        ),
        ClassifierSettings(
            clf=RandomForestClassifier,
            module_kwargs={
                'n_estimators': 100,
                'n_jobs': -1,
            },
            fit_kwargs={},
            enable=True,
        ),
        ClassifierSettings(
            clf=GBDTClassifier,
            module_kwargs={},
            fit_kwargs={},
            enable=True,
        ),
        ClassifierSettings(
            clf=GaussianNB,
            module_kwargs={},
            fit_kwargs={},
            enable=True,
        ),
        ClassifierSettings(
            clf=DecisionTreeClassifier,
            module_kwargs={},
            fit_kwargs={},
            enable=True,
        ),
        ClassifierSettings(
            clf=KNeighborsClassifier,
            module_kwargs={
                'n_jobs': -1,
            },
            fit_kwargs={},
            enable=True,
        ),
    ]

    tester = ClassifierTester(dg)

    #pool = Pool(processes=1)
    #pool.map(lambda x: tester.test_module(x.clf, module_kwargs=x.module_kwargs, fit_kwargs=x.fit_kwargs),
    #         filter(lambda x: x.enable, classifiers))

    for classifier in filter(lambda x: x.enable, classifiers):
        tester.cv_test_module(
            classifier.clf,
            module_kwargs=classifier.module_kwargs,
            fit_kwargs=classifier.fit_kwargs
        )

def predict_all():
    logging.basicConfig(format="%(asctime)s %(levelname)8s: %(message)s", level=logging.DEBUG)

    dg = DataGenerator(
        vectors_file='/mnt/d/Downloads/embedding.vec.json',
        # vectors_file='embedding.vec.json',
        entities_file='drugs.txt',
        dataset_df=pd.read_excel('41467_2019_9186_MOESM5_ESM.xlsx', skipfooter=1),
    )

    clf = GaussianNB()
    clf.fit(*dg.get_all())

    data = tuple(filter(lambda x: isinstance(x, str), dg.ent_dict.keys()))
    comb = combinations(data, 2)
    results = []

    #tees = tee(tqdm(comb, total=len(data)**2//2), cpu_count())
    #pool = Pool(processes=None)

    #pool.map(worker_fn, tees)

    for curr in tqdm(comb, total=len(data)**2//2):
        res = clf.predict_prob(next(dg.embed([curr])).reshape(-1, 1000))
        results.append(curr + tuple(*res.tolist()))

    df = pd.DataFrame(results, columns=['Interaction_A(DrugBank_ID)',
                                        'Interaction_B(DrugBank_ID)',
                                        'Negative Probabilities',
                                        'Positive Probabilities',
                                        ])
    df.sort_values(['Positive Probabilities', 'Negative Probabilities'], ascending=[False, True], inplace=True)
    df.to_hdf('output.h5', 'combination')


def perm_to_comb_1():
    df = pd.read_hdf('output.h5', 'permutation')
    origin = set(map(tuple, pd.read_excel('41467_2019_9186_MOESM5_ESM.xlsx', skipfooter=1).apply(
        lambda x: x.str.strip()).drop_duplicates().to_numpy().tolist()))

    seen = {}
    results = []
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        curr = row[1:3]
        if curr in origin or curr[::-1] in origin:
            continue
        if curr in seen:
            t = seen[curr]
            t.append(row[0])
            result = pd.concat(
                [df.iloc[t[0]][['Interaction_A(DrugBank_ID)', 'Interaction_B(DrugBank_ID)']],
                 df.iloc[t].mean(numeric_only=True)])
            results.append(result)
        elif curr[::-1] in seen:
            t = seen[curr[::-1]]
            t.append(row[0])
            result = pd.concat(
                [df.iloc[t[0]][['Interaction_A(DrugBank_ID)', 'Interaction_B(DrugBank_ID)']],
                 df.iloc[t].mean(numeric_only=True)])
            results.append(result)
        else:
            seen[curr] = [row[0]]
    final = pd.concat(results, axis=1).T
    final.to_hdf('output.h5', 'combination')

def perm_to_comb_dask():
    def dask_worker(df):
        res = df.mean(numeric_only=True)
        res['Comb Cnt'] = df.shape[0]
        return res

    perm = pd.read_hdf('output.h5', 'permutation')
    perm = perm[(perm['Positive Probabilities'] > 0.9999) & (perm['Negative Probabilities'] < 0.0001)]
    perm['Comb Cnt'] = 0
    ## TEST ONLY
    #perm_paired = pd.concat([perm[(perm['Interaction_A(DrugBank_ID)'] == row[2]) & (perm['Interaction_B(DrugBank_ID)'] == row[1])] for row in islice(perm.itertuples(), 0, 10)], axis=0)
    #perm = pd.concat([perm[:10], perm_paired], axis=0)

    origin = set(map(frozenset, pd.read_excel('41467_2019_9186_MOESM5_ESM.xlsx', skipfooter=1).apply(
        lambda x: x.str.strip()).drop_duplicates().to_numpy().tolist()))
    perm_dd = dd.from_pandas(perm, npartitions=cpu_count())
    comb_set = perm_dd.apply(
        lambda row: frozenset((row['Interaction_A(DrugBank_ID)'], row['Interaction_B(DrugBank_ID)'])),
        axis=1,
        meta=('Combination', object)
    )
    perm_dd = dd.concat([perm_dd, comb_set], axis=1)
    comb_dd = perm_dd.groupby(['Combination']).apply(
        dask_worker,
        meta = [
            ('Negative Probabilities', np.float64),
            ('Positive Probabilities', np.float64),
            ('Comb Cnt', int),
        ]
    )
    with ProgressBar():
        results = comb_dd.compute(scheduler='processes')

    results = results.loc[(~results.index.isin(origin)) & (results['Comb Cnt'] > 1)]
    results.drop(columns=['Comb Cnt'], inplace=True)
    results.sort_values(['Positive Probabilities', 'Negative Probabilities'], ascending=[False, True], inplace=True)

    results.to_hdf('output.h5', 'combination')




if __name__ == '__main__':
    perm_to_comb_dask()

