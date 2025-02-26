from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, override, TypeAlias
from inspect import signature

import torch
from brain.util import only_kwargs_from_fn

from zipfile import ZipFile
from requests import get
from pathlib import Path
from semver import Version
import shutil
import os

from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from inspect import isclass, isfunction

def to_version(version: str | Version):
    return version if isinstance(version, Version) else Version.parse(version)


class Data(ABC):
    """ 
        Abstract class that manages the installation, preprocessing, 
        uninstallation, and the retrieval of the training, and testing
        data to be used by the supervisor.

        The `install` and `training_data` methods must be implemented; the
        rest of the functions are overridable - returning None for all but the
        uninstall method, which removes the data-dir, by default.

        `install` installs the data on the host; if the data is in-memory just
        pass/return in the method implementation, or create the data and save it to self for the dataset
        functions to fetch it.

        `training_data` creates the dataset instance.

        `testing_data` returns None, by default; override 
         it if a testing set should be used by the supervisor.

        `uninstall` defaults to deleting the data-dir; override it if their are
        side effects like maybe in database cases, or if the data-dir shares
        other data that should not be deleted.

        `is_installed` is overridable, it defaults to checking to see if the
        data-dir is empty.

        `__preprocess__` is called after the install method in the supervisor, only
        if the data is not already installed; it does nothing and is overridable. Use
        it for eg extracting zip data and any mutations to the data before the dataset
        instance is used to fetch it.

        `uninstall` is optionally called by the user after the supervisor runs.
    """
    def __init__(
        self, 
        name: str,
        version: str | Version,
        data_dir: str | Path | None = None
    ):
        self.name = name

        self.version = to_version(version)

        self.data_dir = data_dir


    @abstractmethod
    def install(
        self
    ):
        pass


    @override
    def preprocess(
        self
    ):
        pass


    @override
    def uninstall(
        self
    ):
        if self.data_dir is not None:
            shutil.rmtree(self.data_dir)


    @abstractmethod
    def training_data(
        self
    ) -> Dataset:
        pass


    @override
    def testing_data(
        self
    ) -> Dataset | None:
        return None


    @override
    def is_installed(
            self
    ):
        return len(os.listdir(self.data_dir)) > 0

Class: TypeAlias = Callable

Function: TypeAlias = Callable

class Model:
    def __init__(
        self,
        name: str,
        version: str | Version,
        module: Module
    ):
        self.name = name

        self.version = to_version(version)

        self.module = module

class Metrics:
    def __init__(
        self,
        interval: int,
        param_norm: bool = False,
        grad_norm: bool = False
    ):
        self.metrics = {}

        self.interval = interval

        self.has = {
            "param_norm": param_norm, 
            "grad_norm": grad_norm
        }

        self.metrics["loss_epoch"] = None

        self.metrics["loss_interval"] = None

        self.metrics["loss_running"] = None

        self.metrics["test_loss_epoch"] = None

        self.metrics["test_loss_interval"] = None

        self.metrics["test_loss_running"] = None

        if param_norm:
            self.metrics["param_norm_epoch"] = None

            self.metrics["param_norm_interval"] = None

            self.metrics["param_norm_running"] = None
        
        if grad_norm:
            self.metrics["grad_norm_epoch"] = None

            self.metrics["grad_norm_interval"] = None

            self.metrics["grad_norm_running"] = None
        

        def is_tracking(metric: str):
            return self.has[metric]
        

        def __getitem__(self, key) -> Any:
            return self.metrics[key]
        

        def __setitem__(self, key, value):
            self.metrics[key] = value


class Brain:
    name = None

    version = None

    model_name = None

    model_version = None

    data_name = None

    data_version = None

    supervisor_name = None

    supervisor_version = None

    metrics = None

    def __init__(
        self,
        name: str,
        version: str | Version,
        model: Model,
    ):
        self.name = name

        self.version = to_version(version)

        self.model = model

        self.model_name = model.name

        self.model_version = model.version


class Supervisor:
    """
        A supervisor is an optimizer and a loss function with
        the hyperparameters to construct them.

        The supervisor trains a model on a dataset using
        a `Data` subclassed instance, an optimizer class, a loss
        function or class, and the hyperparameters to construct
        the optimizer, the loss (if given as a class), the
        data-loaders, as well as the metrics tracking and debugging
        options.

        The hyperparameters must contain the keyword arguments 
        in the constructors of the optimizer class, the loss 
        function class (if it is a class), and the `DataLoader`. 
        Further metrics tracking and debugging options can be
        provided.

        Using `supervise` creates a trained model on the datasets
        from the `Data` subclass.

        So, a supervisor is a snapshot of the optimizer, loss, 
        and hyperparameters that can train models on data.
    """
    def __init__(
            self, 
            name: str, 
            version: str | Version,
            optimizer: Class,
            loss: Class | Function,
            **hyperparameters
    ):
        self.name = name

        self.version = to_version(version)

        self.optimizer = optimizer

        self.loss = loss

        self.hyperparameters = hyperparameters


    def supervise(
            self, 
            name: str,
            version: str | Version,
            model: Model,
            data: Data,
            epochs: int
    ) -> Brain:
        brain = Brain(name, version, model)

        brain.supervisor_name = self.name

        brain.supervisor_version = self.version

        brain.data_name = data.name

        brain.data_version = data.version

        brain.metrics = Metrics(
            **only_kwargs_from_fn(
                Metrics.__init__, 
                self.hyperparameters
            )
        )

        optim = self.optimizer.__init__(
            **only_kwargs_from_fn(
                self.optimizer.__init__, self.hyperparameters
            )
        )

        loss_fn = self.loss.__init__(
            **only_kwargs_from_fn(
                self.loss.__init__,
                self.hyperparameters
            )
        ) if isclass(self.loss) else self.loss

        if not data.is_installed():
            data.install()

            data.preprocess()

        training_data = data.training_data()

        testing_data = data.testing_data()

        has_test = testing_data is not None

        train_dl = DataLoader(
            dataset=training_data,
            **only_kwargs_from_fn(
                DataLoader.__init__,
                self.hyperparameters
            )
        )

        test_dl = None if testing_data is None else DataLoader(
            dataset=testing_data,
            **only_kwargs_from_fn(
                DataLoader.__init__,
                self.hyperparameters
            )
        )

        epoch_losses = []

        running_losses = []

        interval_losses = []

        interval_loss = 0

        running_loss = 0

        if test_dl is not None:
            epoch_test_losses = []

            interval_test_losses = []

            running_test_losses = []

            interval_test_loss = 0

            running_test_loss = 0

        param_norms_epoch = []

        param_norms_interval = []

        param_norms_running = []

        param_norm_interval = 0
        
        param_norm_running = 0

        grad_norms_epoch = []

        grad_norms_interval = []

        grad_norms_running = []

        grad_norm_interval = 0

        grad_norm_running = 0
        
        nn: Module = brain.model.module

        for epoch in epochs:
            nn.train(True)

            for features_batch, targets_batch in train_dl:
                optim.zero_grad()

                predictions = nn(features_batch)

                loss = loss_fn(predictions, targets_batch)

                loss.backward()

                optim.step()

                loss = loss.item()

                interval_loss += loss

                running_loss += loss
            
            # FIXME: all the loss metrics
            if (epoch + 1) % brain.metrics.interval == 0:
                epoch_losses.append(loss)

                interval_losses.append(
                    interval_loss / brain.metrics.interval
                )

                interval_loss = 0

                running_losses.append(
                    running_loss / epoch
                )

                print(loss)

                # FIXME: parameter norms
                if brain.metrics.is_tracking("grad_norm"):
                    pass
                    
                # FIXME: gradient norms
                if brain.metrics.is_tracking("param_norm"):
                    pass
        
        if test_dl is not None:
            with torch.inference_mode():
                nn.eval()

                for test_features_batch, test_targets_batch in test_dl:
                    test_predictions = nn(test_features_batch)

                    test_loss = loss_fn(
                        test_predictions,
                        test_targets_batch
                    ).item()

                    interval_test_loss += test_loss

                    running_test_loss += test_loss

                    # FIXME: all the test metrics
                    if (epoch + 1) % brain.metrics.interval == 0:
                        epoch_test_losses.append(test_loss.item())

                        interval_test_loss = 0