from .evaluator import ContSurv_Evaluator
from .evaluator import DiscSurv_Evaluator
from .evaluator import CoxSurv_Evaluator

def prepare_evaluator(output_type, **kws):
    assert output_type in ['continuous', 'discrete', 'prohazard']
    if output_type == 'continuous':
        evaluator = ContSurv_Evaluator(**kws)
    elif output_type == 'discrete':
        evaluator = DiscSurv_Evaluator(**kws)
    elif output_type == 'prohazard':
        evaluator = CoxSurv_Evaluator(**kws)
    else:
        evaluator = None
    return evaluator
