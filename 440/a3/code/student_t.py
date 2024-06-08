import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.chi2 import Chi2
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import (
    _batch_mv,
    _batch_mahalanobis,
    _precision_to_scale_tril,
    MultivariateNormal,
)
from torch.distributions.utils import _standard_normal, lazy_property


LOG_PI = math.log(math.pi)


class MultivariateT(Distribution):
    """
    A multivariate t distribution; see
    https://en.wikipedia.org/wiki/Multivariate_t-distribution

    Based on torch.distributions.MultivariateNormal,
    but copy-pastes instead of inheriting because
    (a) semantically it's the other way: normal is a special case of t as df -> oo;
    (b) too much code is annoying if you try it that way.
    """

    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        df,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        if isinstance(df, Number):
            df = torch.tensor(df, dtype=loc.dtype, device=loc.device)

        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix "
                "or scale_tril may be specified."
            )

        for mat, name in [
            (scale_tril, "scale_tril"),
            (covariance_matrix, "covariance_matrix"),
            (precision_matrix, "precision_matrix"),
        ]:
            if mat is not None:
                if mat.dim() < 2:
                    raise ValueError(
                        f"{name} matrix must be at least two-dimensional, "
                        "with optional leading batch dimensions"
                    )
                batch_shape = torch.broadcast_shapes(
                    mat.shape[:-2], loc.shape[:-1], df.shape
                )
                setattr(self, name, mat.expand(batch_shape + (-1, -1)))
                break

        self.df = df.expand(batch_shape)
        self.loc = loc.expand(batch_shape + (-1,))

        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        df_shape = batch_shape
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.df = self.df.expand(df_shape)
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateT, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(
            self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        if (self.df > 1).all():
            return self.loc
        else:
            raise ValueError("MultivariateT has no mean if df <= 1")

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        if (self.df > 2).all():
            scale = self.df / (self.df - 2)
            sigma = (
                self._unbroadcasted_scale_tril.pow(2)
                .sum(-1)
                .expand(self._batch_shape + self._event_shape)
            )
            return scale * sigma
        else:
            raise ValueError(f"MultivariateT has no variance if df {df} <= 2")

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        u = Chi2(self.df).rsample(shape)

        # y ~ N(0, Sigma)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        y = _batch_mv(self._unbroadcasted_scale_tril, eps)

        return torch.sqrt(self.df / u) * y + self.loc

    def log_prob(self, value):
        value = torch.as_tensor(
            value
        )  # not in torch version, but we might want numpy here
        if self._validate_args:
            self._validate_sample(value)

        p = self.loc.shape[-1]
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, value - self.loc)

        log_const = (
            torch.lgamma((self.df + p) / 2)
            - torch.lgamma(self.df / 2)
            - p / 2 * (torch.log(self.df) + LOG_PI)
            - half_log_det
        )
        return log_const - (self.df + p) / 2 * torch.log1p(M / self.df)

    @classmethod
    def mle(
        cls,
        X,
        init_df=None,
        init_mu=None,
        init_sigma=None,
        max_iter=10_000,
        lr=0.1,
        **kwargs,
    ):
        X = torch.as_tensor(X)
        n, d = X.shape

        params = []
        # start with moment matching and random df > 2 (so variance exists)
        if init_df is None:
            init_df = 2 + 10 * torch.exp(
                torch.randn((), device=X.device, dtype=X.dtype)
            )
        if init_mu is None:
            init_mu = X.mean(0)
        if init_sigma is None:
            init_sigma = torch.cov(X.t()) * ((init_df - 2) / init_df)

        init_L = torch.linalg.cholesky(init_sigma)
        tril_inds = tuple(torch.tril_indices(d, d, offset=-1))

        params = [
            a.detach()
            .to(device=X.device, dtype=X.dtype, copy=True)
            .requires_grad_(True)
            for a in [
                torch.log(init_df),  # log(df); ensures it remains positive
                init_mu,  # mean can be anything
                init_L.diagonal().log(),  # cholesky diagonal must be positive
                init_L[tril_inds],  # cholesky lower triangle can be anything
            ]
        ]

        def get_dist(validate_args=None):
            df = params[0].exp()
            L = torch.zeros(d, d, dtype=params[2].dtype, device=params[2].device)
            L[range(d), range(d)] = params[2].exp()
            L[tril_inds] = params[3]
            return cls(df=df, loc=params[1], scale_tril=L, validate_args=validate_args)

        opt = torch.optim.LBFGS(params, max_iter=max_iter, lr=lr, **kwargs)

        def closure():
            opt.zero_grad()
            loss = -get_dist(validate_args=False).log_prob(X).sum()
            loss.backward()
            return loss

        opt.step(closure)  # actually runs lots of steps

        return get_dist(validate_args=True)
