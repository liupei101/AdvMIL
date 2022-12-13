from .evaluator import ContSurv_Evaluator
from .evaluator import DiscSurv_Evaluator

def prepare_evaluator(output_type, **kws):
	assert output_type in ['continuous', 'discrete']
	if output_type == 'continuous':
		evaluator = ContSurv_Evaluator(**kws)
	elif output_type == 'discrete':
		evaluator = DiscSurv_Evaluator(**kws)
	else:
		evaluator = None
	return evaluator
