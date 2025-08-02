import pymc as pm
import numpy as np
from scipy.special import expit
from app.utils import conf_manager


def get_config_params():

    custom_params = conf_manager.get_value("custom_parameters")
    if custom_params is None:
        raise ValueError("custom_parameters not found in config or is None")

    def parse_vector(val):
        if val is None:
            return np.array([])
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                return np.array([])
            return np.array([float(x) for x in val.split(",")])
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.array(val)
        return np.array([val])

    for key in ["alpha_vector", "p_vector", "weights", "means", "sigmas"]:
        if key in custom_params:
            custom_params[key] = parse_vector(custom_params[key])

    distribution_params: dict = {
        "Normal": {
            "mu": custom_params.get("mean", 0),
            "sigma": custom_params.get("sigma", 1),
        },
        "HalfNormal": {"sigma": custom_params.get("sigma", 1)},
        "Cauchy": {
            "alpha": custom_params.get("alpha", 1),
            "beta": custom_params.get("beta", 1),
        },
        "Exponential": {"lam": custom_params.get("lambda", 1)},
        "Beta": {
            "alpha": custom_params.get("alpha", 1),
            "beta": custom_params.get("beta", 1),
        },
        "ChiSquared": {"nu": custom_params.get("nu", 1)},
        "ExGaussian": {
            "mu": custom_params.get("mu", 0),
            "sigma": custom_params.get("sigma", 1),
            "nu": custom_params.get("nu", 1),
        },
        "Gamma": {
            "alpha": custom_params.get("alpha", 1),
            "beta": custom_params.get("beta", 1),
        },
        "Logistic": {
            "mu": custom_params.get("mu", 0),
            "s": custom_params.get("scale", 1),
        },
        "LogNormal": {
            "mu": custom_params.get("mu", 0),
            "sigma": custom_params.get("sigma", 1),
        },
        "Uniform": {
            "lower": custom_params.get("lower", 0),
            "upper": custom_params.get("upper", 1),
        },
        "Weibull": {
            "alpha": custom_params.get("alpha", 1),
            "beta": custom_params.get("beta", 1),
        },
        "Bernoulli": {"p": custom_params.get("p", 0.5)},
        "Binomial": {"n": custom_params.get("n", 1), "p": custom_params.get("p", 0.5)},
        "BetaBinomial": {
            "alpha": custom_params.get("alpha", 1),
            "beta": custom_params.get("beta", 1),
            "n": custom_params.get("n", 1),
        },
        "Categorical": {"p": custom_params.get("p", 0.5)},
        "Poisson": {"mu": custom_params.get("mu", 1)},
        "Dirichlet": {"a": custom_params.get("alpha_vector", np.array([1.0]))},
        "DirichletMultinomial": {
            "a": custom_params.get("alpha_vector", np.array([1.0])),
            "n": custom_params.get("n", 1),
        },
        "Multinomial": {
            "n": custom_params.get("n", 1),
            "p": custom_params.get("p_vector", np.array([1.0])),
        },
        "NormalMixture": {
            "w": custom_params.get("weights", np.array([1.0])),
            "mus": custom_params.get("means", np.array([0.0])),
            "sigmas": custom_params.get("sigmas", np.array([1.0])),
        },
        "GaussianRandomWalk": {"sigma": custom_params.get("sigma", 1)},
        "AR1": {
            "k": custom_params.get("k", 0),
            "tau": custom_params.get("tau", 1),
            "rho": custom_params.get("rho", 0),
        },
    }

    if not distribution_params:
        raise ValueError("No distribution parameters found in config")

    distribution = conf_manager.get_value("distribution")
    if distribution is None:
        raise ValueError("distribution not found in config or is None")
    if distribution not in distribution_params:
        raise ValueError(f"Distribution '{distribution}' not supported.")

    params = distribution_params[distribution]

    if distribution == "Dirichlet":
        a = params["a"]
        if not isinstance(a, np.ndarray) or a.ndim != 1 or np.any(a <= 0):
            raise ValueError(
                "Dirichlet 'a' parameter must be a 1D array of positive floats."
            )

    return params


def apply_pymc_adjustment(preds: np.ndarray) -> np.ndarray:
    """
    Ajusta las predicciones con PyMC de forma genérica usando la distribución seleccionada.
    Soporta binario (1D) o multiclass (2D) aplicando logistic o softmax.
    """

    preds = np.array(preds)
    params = get_config_params()
    distribution = conf_manager.get_value("distribution")
    if distribution is None:
        raise ValueError("No distribution specified in config.")

    # Mapeo de distribuciones
    distributions = {
        "Normal": lambda name, params, shape: pm.Normal(
            name, mu=params["mu"], sigma=params["sigma"], shape=shape
        ),
        "HalfNormal": lambda name, params, shape: pm.HalfNormal(
            name, sigma=params["sigma"], shape=shape
        ),
        "Cauchy": lambda name, params, shape: pm.Cauchy(
            name, alpha=params["alpha"], beta=params["beta"], shape=shape
        ),
        "Exponential": lambda name, params, shape: pm.Exponential(
            name, lam=params["lam"], shape=shape
        ),
        "Beta": lambda name, params, shape: pm.Beta(
            name, alpha=params["alpha"], beta=params["beta"], shape=shape
        ),
        "ChiSquared": lambda name, params, shape: pm.ChiSquared(
            name, nu=params["nu"], shape=shape
        ),
        "ExGaussian": lambda name, params, shape: pm.ExGaussian(
            name, mu=params["mu"], sigma=params["sigma"], nu=params["nu"], shape=shape
        ),
        "Gamma": lambda name, params, shape: pm.Gamma(
            name, alpha=params["alpha"], beta=params["beta"], shape=shape
        ),
        "Logistic": lambda name, params, shape: pm.Logistic(
            name, mu=params["mu"], s=params["s"], shape=shape
        ),
        "LogNormal": lambda name, params, shape: pm.LogNormal(
            name, mu=params["mu"], sigma=params["sigma"], shape=shape
        ),
        "Uniform": lambda name, params, shape: pm.Uniform(
            name, lower=params["lower"], upper=params["upper"], shape=shape
        ),
        "Weibull": lambda name, params, shape: pm.Weibull(
            name, alpha=params["alpha"], beta=params["beta"], shape=shape
        ),
        "Bernoulli": lambda name, params, shape: pm.Bernoulli(
            name, p=params["p"], shape=shape
        ),
        "Binomial": lambda name, params, shape: pm.Binomial(
            name, n=params["n"], p=params["p"], shape=shape
        ),
        "BetaBinomial": lambda name, params, shape: pm.BetaBinomial(
            name, alpha=params["alpha"], beta=params["beta"], n=params["n"], shape=shape
        ),
        "Categorical": lambda name, params, shape: pm.Categorical(
            name, p=params["p"], shape=shape
        ),
        "Poisson": lambda name, params, shape: pm.Poisson(
            name, mu=params["mu"], shape=shape
        ),
        "Dirichlet": lambda name, params, shape: pm.Dirichlet(
            name, a=params["a"], shape=shape
        ),
        "DirichletMultinomial": lambda name, params, shape: pm.DirichletMultinomial(
            name, a=params["a"], n=params["n"], shape=shape
        ),
        "Multinomial": lambda name, params, shape: pm.Multinomial(
            name, n=params["n"], p=params["p"], shape=shape
        ),
        "NormalMixture": lambda name, params, shape: pm.Mixture(
            name,
            w=params["w"],  # weights
            comp_dists=[
                pm.Normal.dist(mu=mu, sigma=sigma, shape=shape)
                for mu, sigma in zip(params["mus"], params["sigmas"])
            ],
            shape=shape,
        ),
        "GaussianRandomWalk": lambda name, params, shape: pm.GaussianRandomWalk(
            name, sigma=params["sigma"], shape=shape
        ),
        "AR1": lambda name, params, shape: pm.AR(
            name, k=params["k"], tau=params["tau"], rho=params["rho"], shape=shape
        ),
    }

    is_multiclass = preds.ndim > 1
    shape = preds.shape[1] if is_multiclass else None

    dist_func = distributions.get(distribution)
    if dist_func is None:
        raise ValueError(f"Distribución '{distribution}' no soportada.")

    with pm.Model() as _:

        adjustment = dist_func("adjustment", params, shape=shape)
        adjusted_logits = preds + adjustment

        if conf_manager.get_value("markov"):
            print("Using NUTS sampler for Markov adjustment")
            pm.NUTS()
        else:
            print("Using ADVI sampler for adjustment")
            pm.fit(n=1000, method="advi", progressbar=False)

        trace = pm.sample(500, tune=500, chains=2, cores=4, progressbar=False)

        if is_multiclass:
            adjustment_samples = (
                trace.posterior["adjustment"].mean(dim=("chain", "draw")).values  # type: ignore
            )
            adjusted_logits = preds + adjustment_samples
            exp_vals = np.exp(adjusted_logits)
            adjusted_preds = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            adjustment_samples = trace.posterior["adjustment"].mean().values  # type: ignore
            adjusted_logits = preds + adjustment_samples
            adjusted_preds = expit(adjusted_logits)

    return adjusted_preds


def custom_objective_factory():
    """
    Crea una función de objetivo personalizada con la distribución especificada.
    """

    def custom_objective(preds, dtrain):
        adjusted_preds = apply_pymc_adjustment(preds)
        labels = dtrain.get_label()

        if preds.ndim == 1:
            # Binario
            grad = adjusted_preds - labels
            hess = adjusted_preds * (1 - adjusted_preds)
        else:
            # Multiclass (aprox. de grad/hess)
            rows = len(labels)
            grad = np.copy(adjusted_preds)
            grad[np.arange(rows), labels.astype(int)] -= 1.0
            hess = adjusted_preds * (1.0 - adjusted_preds)
        return grad, hess

    return custom_objective
