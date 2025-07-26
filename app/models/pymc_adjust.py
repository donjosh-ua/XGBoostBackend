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
            return None
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                return None
            return np.array([float(x) for x in val.split(",")])
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.array(val)
        return val

    for key in ["alpha_vector", "p_vector", "weights", "means", "sigmas"]:
        if key in custom_params:
            custom_params[key] = parse_vector(custom_params[key])

    distribution_params = {
        "Normal": {"mu": custom_params["mean"], "sigma": custom_params["sigma"]},
        "HalfNormal": {"sigma": custom_params["sigma"]},
        "Cauchy": {"alpha": custom_params["alpha"], "beta": custom_params["beta"]},
        "Exponential": {"lam": custom_params["lambda"]},
        "Beta": {"alpha": custom_params["alpha"], "beta": custom_params["beta"]},
        "ChiSquared": {"nu": custom_params["nu"]},
        "ExGaussian": {
            "mu": custom_params["mu"],
            "sigma": custom_params["sigma"],
            "nu": custom_params["nu"],
        },
        "Gamma": {"alpha": custom_params["alpha"], "beta": custom_params["beta"]},
        "Logistic": {"mu": custom_params["mu"], "s": custom_params["scale"]},
        "LogNormal": {"mu": custom_params["mu"], "sigma": custom_params["sigma"]},
        "Uniform": {"lower": custom_params["lower"], "upper": custom_params["upper"]},
        "Weibull": {"alpha": custom_params["alpha"], "beta": custom_params["beta"]},
        "Bernoulli": {"p": custom_params["p"]},
        "Binomial": {"n": custom_params["n"], "p": custom_params["p"]},
        "BetaBinomial": {
            "alpha": custom_params["alpha"],
            "beta": custom_params["beta"],
            "n": custom_params["n"],
        },
        "Categorical": {"p": custom_params["p"]},
        "Poisson": {"mu": custom_params["mu"]},
        "Dirichlet": {"a": custom_params["alpha_vector"]},
        "DirichletMultinomial": {
            "a": custom_params["alpha_vector"],
            "n": custom_params["n"],
        },
        "Multinomial": {"n": custom_params["n"], "p": custom_params["p_vector"]},
        "NormalMixture": {
            "w": custom_params["weights"],
            "mus": custom_params["means"],
            "sigmas": custom_params["sigmas"],
        },
        "GaussianRandomWalk": {"sigma": custom_params["sigma"]},
        "AR1": {
            "k": custom_params["k"],
            "tau": custom_params["tau"],
            "rho": custom_params["rho"],
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

    return params


def apply_pymc_adjustment(preds: np.ndarray) -> np.ndarray:
    """
    Ajusta las predicciones con PyMC de forma genérica usando la distribución seleccionada.
    Soporta binario (1D) o multiclass (2D) aplicando logistic o softmax.
    """
    preds = np.array(preds)
    params = get_config_params()
    distribution = conf_manager.get_value("distribution")

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

    # Detección de binario o multiclass
    is_multiclass = preds.ndim > 1
    shape = preds.shape[1] if is_multiclass else None

    dist_func = distributions.get(distribution)
    if dist_func is None:
        raise ValueError(f"Distribución '{distribution}' no soportada.")

    with pm.Model() as _:

        adjustment = dist_func("adjustment", params, shape=shape)
        adjusted_logits = preds + adjustment

        trace = pm.sample(500, tune=500, chains=2, cores=4, progressbar=False)

        if is_multiclass:
            adjustment_samples = (
                trace.posterior["adjustment"].mean(dim=("chain", "draw")).values
            )
            adjusted_logits = preds + adjustment_samples
            exp_vals = np.exp(adjusted_logits)
            adjusted_preds = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            adjustment_samples = trace.posterior["adjustment"].mean().values
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
