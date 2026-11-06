import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	tpos = (y_true * y_pred).sum()
	fpos = ((1 - y_true) * y_pred).sum()
	fneg = ((1 - y_pred) * y_true).sum()

	prec = tpos/(tpos + fpos)
	recl = tpos/(tpos + fneg)

	return round(((1 + beta**2) * prec * recl)/(beta**2 * prec + recl), 3)
