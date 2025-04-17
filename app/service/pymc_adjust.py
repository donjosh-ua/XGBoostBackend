import pymc as pm
import numpy as np
from scipy.special import expit
from app.config import conf_manager


# Configuracion de la distribución
def get_config_params():

    custom_params = conf_manager.get_value("custom_parameters")
    distribution_params = {
        "Normal": {"mu": custom_params["mean"], "sigma": custom_params["sigma"]},
        "HalfNormal": {"sigma": custom_params["sigma"]},
        "Cauchy": {"alpha": custom_params["alpha"], "beta": custom_params["beta"]},
        "Exponential": {"lam": custom_params["lambda"]},
    }

    params = distribution_params[conf_manager.get_value("distribution")]

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
    }

    # Detección de binario o multiclass
    is_multiclass = preds.ndim > 1
    shape = preds.shape[1] if is_multiclass else None

    dist_func = distributions.get(distribution)
    if dist_func is None:
        raise ValueError(f"Distribución '{distribution}' no soportada.")

    with pm.Model() as model:

        adjustment = dist_func("adjustment", params, shape=shape)
        adjusted_logits = preds + adjustment

        trace = pm.sample(1000, tune=1000, chains=4, cores=4, progressbar=True)
        # trace = pm.sample(1000, return_inferencedata=False, progressbar=False)

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
