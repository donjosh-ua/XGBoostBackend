import pymc as pm
import numpy as np
from scipy.special import expit

# Constantes
kSeed = 1994
kRatio = 0.7
num_rounds = 50
np.random.seed(kSeed)

# Archivo de datos
datafile = 'pima-indians-diabetes.data.csv'
# datafile = 'glass.csv'

# Configuracion de la distribución
distribution = 'Normal'
distribution_params = {
    'Normal': {'mu': 0, 'sigma': 10},
    'HalfNormal': {'sigma': 10},
    'Cauchy': {'alpha': 0, 'beta': 1},
    'Exponential': {'lam': 1}
}


def get_config_params(distribution: str):
    params = distribution_params[distribution]
    return params

def apply_pymc_adjustment(preds: np.ndarray) -> np.ndarray:
    """
    Ajusta las predicciones con PyMC de forma genérica usando la distribución seleccionada.
    Soporta binario (1D) o multiclass (2D) aplicando logistic o softmax.
    """
    preds = np.array(preds)
    params = get_config_params(distribution)

    # Mapeo de distribuciones
    distributions = {
        'Normal': lambda name, params, shape: pm.Normal(
            name, mu=params['mu'], sigma=params['sigma'], shape=shape
        ),
        'HalfNormal': lambda name, params, shape: pm.HalfNormal(
            name, sigma=params['sigma'], shape=shape
        ),
        'Cauchy': lambda name, params, shape: pm.Cauchy(
            name, alpha=params['alpha'], beta=params['beta'], shape=shape
        ),
        'Exponential': lambda name, params, shape: pm.Exponential(
            name, lam=params['lam'], shape=shape
        )
    }

    # Detección de binario o multiclass
    is_multiclass = (preds.ndim > 1)
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
            adjustment_samples = trace.posterior['adjustment'].mean(dim=("chain", "draw")).values
            adjusted_logits = preds + adjustment_samples
            exp_vals = np.exp(adjusted_logits)
            adjusted_preds = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            adjustment_samples = trace.posterior['adjustment'].mean().values
            adjusted_logits = preds + adjustment_samples
            adjusted_preds = expit(adjusted_logits)

    return adjusted_preds



def custom_objective_factory(distribution: str):
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
