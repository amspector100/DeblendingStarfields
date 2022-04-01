import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

mag_vec = np.concatenate((np.array([15]), 
						  np.arange(19, 22.2, 0.4), 
						  np.array([100])))

MAG_CONST = 2.5
def convert_nmgy_to_mag(nmgy):
	return 22.5 - MAG_CONST * np.log10(nmgy)

def filter_params(locs, fluxes, slen, pad = 5):
	"""
	Remove sources too close to the border of the image.
	"""
	assert len(locs.shape) == 2

	if fluxes is not None:
		assert len(fluxes.shape) == 1
		assert len(fluxes) == len(locs)

	_locs = locs * (slen - 1)
	which_params = (_locs[:, 0] >= pad) & (_locs[:, 0] <= (slen - pad)) & \
						(_locs[:, 1] >= pad) & (_locs[:, 1] <= (slen - pad))

	if fluxes is not None:
		return locs[which_params], fluxes[which_params], which_params
	else:
		return locs[which_params], None, which_params

def get_locs_error(locs, true_locs):
	"""
	Parameters
	----------
	locs : (n, 2) shaped array of estimated locations
	true_locs : (m, 2) shaped array of true locations
	
	Returns
	-------
	(m, n)-shaped array of taxicab distance between
	each estimated location and the true location.
	"""
	n = locs.shape[0]
	return np.sqrt(np.power(
		locs.reshape(1, -1, 2) - true_locs.reshape(-1, 1, 2), 2
	).sum(axis=2))


def get_mag_error(mags, true_mags):
	"""
	Parameters
	----------
	mags : (n,) shaped array of estimated magnitudes
	true_locs : (m,) shaped array of true magnitudes
	
	Returns
	-------
	(m, n)-shaped array of abs distance between log mags
	and log true mags.
	"""
	return np.abs(
		mags.reshape(1, -1) - \
		true_mags.reshape(-1, 1)
	)

def get_summary_stats(
	est_locs,
	true_locs,
	loc_errors,
	slen,
	nsignal_ci=None,
	nelec_per_nmgy=None,
	est_fluxes=None,
	true_fluxes=None,
	flux_errors=None,
	pad = 5,
	slack = 0.5
):
	"""
	Parameters
	----------
	loc_errors : np.ndarray
		(n,)-length array of radii of bounding boxes for locations
	flux_errors : np.ndarray
		(n,)-length array of radii of bounding interval (in log space)
		for fluxes.
	slen : int
		number of pixels in the (square) image
	nsignal_ci : list of sets
		List of sets specifyng the confidence intervals for the number
		of sources.
	pad : int 
		number of pixels of border to exclude
	slack : float
		expected amount of distance (in pixels) between
		true and estimated sources due to differences
		in imaging.
	"""

	# remove border
	est_locs, est_fluxes, which_params = filter_params(
		est_locs, 
		est_fluxes, 
		slen,
		pad
	)
	loc_errors = loc_errors[which_params]
	if flux_errors is not None:
		flux_errors = flux_errors[which_params]
	
	true_locs, true_fluxes, which_true_locs = filter_params(
		true_locs,
		true_fluxes,
		slen,
		pad
	)

	if (est_fluxes is None) or (true_fluxes is None):
		mag_error = 0.
	else:
		# convert to magnitude
		est_mags = convert_nmgy_to_mag(est_fluxes / nelec_per_nmgy)
		true_mags = convert_nmgy_to_mag(true_fluxes / nelec_per_nmgy)
		mag_error = get_mag_error(est_mags, true_mags)


	# location errors
	locs_error = get_locs_error(est_locs * (slen - 1), true_locs * (slen - 1))
	locs_error_tol = slack + (slen - 1) * loc_errors.reshape(1, -1)
	locs_flags = (locs_error < locs_error_tol)
	
	# mag errors
	if flux_errors is not None:
		mags_error_tol = slack + MAG_CONST * flux_errors.reshape(1, -1)
	else:
		mags_error_tol = np.inf 

	mags_flags = (mag_error < mags_error_tol)


	# array (nsource x nest): is x source contained by y est
	disc_bool = locs_flags * mags_flags

	# array : for each true source, is there a matching estimated source
	tpr_bool = np.any(disc_bool, axis=1)

	if nsignal_ci is None:
		# array: for each estimated source, is there a matching true source
		ppv_bool = np.any(disc_bool, axis=0)
	else:
		# same as above, but accounting for number of sources
		nsignals = np.sum(disc_bool, axis=0).astype(int)
		ppv_bool = np.array([
			int(nsignals[i]) in nsignal_ci[i] for i in range(nsignals.shape[0])
		])

	return tpr_bool, ppv_bool, disc_bool, which_params, which_true_locs

def blip_output_to_catalogue(
	rej_nodes
):
	# Edge case of no rejections
	if len(rej_nodes) == 0:
		return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))

	# Infer dimensionality
	keys = rej_nodes[0].data.keys()
	d = 1
	for key in keys:
		if key[0:3] == 'dim':
			d = max(d, int(key.split("dim")[-1]) + 1)

	# Create estimated locations and errors
	locs_est = np.zeros((len(rej_nodes), d))
	locs_error = np.zeros((len(rej_nodes)))
	locs_peps = np.zeros((len(rej_nodes)))
	weights = np.zeros((len(rej_nodes)))
	for i, node in enumerate(rej_nodes):
		locs_error[i] = node.data['radius']
		for j in range(d):
			locs_est[i, j] = node.data[f'dim{j}']
		locs_peps[i] = node.pep
		weights[i] = node.data['weight']

	return locs_est, locs_error, locs_peps, weights

def disc_bool_to_ndisc(
	disc_bool
):
	m = disc_bool.shape[0] # number of sources
	n = disc_bool.shape[1] # number of estimated sources

	# List of sources/estimates we haven't used yet
	undisc_sources = np.ones(m).astype(bool)
	unused_ests = set(list(range(n)))

	# Iteratively count false/true discoveries and remove
	# elements from the source/estimator list
	true_disc = 0
	false_disc = 0
	while len(unused_ests) > 0:
		# Work with the estimated source which has the fewest
		# number of true discoveries associated with it
		lunused_ests = list(unused_ests)
		true_disc_per_est = np.sum(
			disc_bool[undisc_sources][:, lunused_ests],
			axis=0
		)
		j = np.argmin(true_disc_per_est)
		# convert to other indexing
		gj = lunused_ests[j]
		# Count false discoveries
		if true_disc_per_est[j] == 0:
			false_disc += 1
		# For true discoveries, take a source out of the source set
		# to avoid double-counting
		else:
			source = np.argmax(undisc_sources & disc_bool[:, gj])
			undisc_sources[source] = 0
			true_disc += 1
			
		unused_ests -= set([gj])

	return true_disc, false_disc

def catalogue_power_fdr(
	locs_true,
	locs_est,
	locs_error,
	slen,
	nsignal_ci=None,
	weights=None,
	slack=0.0,
	return_bools=False,
	**kwargs
):
	tpr_bool, ppv_bool, disc_bool, which_ests, which_locs = get_summary_stats(
		est_locs=locs_est,
		true_locs=locs_true,
		loc_errors=locs_error,
		nsignal_ci=nsignal_ci,
		slen=slen,
		slack=slack,
		**kwargs
	)

	# FDR
	if nsignal_ci is None: # for MAP estimates without disjointness
		true_disc, false_disc = disc_bool_to_ndisc(disc_bool)
	else: 
		true_disc = np.sum(ppv_bool)
		false_disc = locs_est.shape[0] - true_disc


	# Naive power
	power = true_disc / max(1, np.sum(which_locs))
	fdr = false_disc / max(1, locs_est.shape[0])
	if locs_est.shape[0] == 0:
		assert fdr == 0
	if weights is None:
		if nsignal_ci is not None:
			res_power = np.sum(ppv_bool / np.array([len(x) for x in nsignal_ci]))
		else:
			res_power = np.sum(ppv_bool / locs_error)
	elif weights == 'const':
		res_power = true_disc
	else:
		res_power = np.sum(ppv_bool * weights)
	if not return_bools:
		return power, fdr, res_power
	else:
		return power, fdr, res_power, ppv_bool

def plot_rejections_matplotlib(
	all_ests,
	all_errors,
	all_peps,
	locs_true,
	slen,
	est_names=None,
	image=None
):
	# Number of methods
	n_methods = len(all_ests)
	if est_names is None:
		est_names = [f'Est {j}' for j in range(n_methods)]
		
	scale = slen - 1

	# Create overarching figure
	ncols = int(np.ceil(n_methods / 2))
	fig, axarr = plt.subplots(2, ncols, figsize=(10, 10), sharey=True)
	for i in range(n_methods):
		irow = i // ncols
		icol = i % ncols
		axarr[irow, icol].matshow(image[0, 0], cmap=plt.cm.gray)
		axarr[irow, icol].set_yticks([])
		axarr[irow, icol].set_xticks([])
		# Plot truth
		axarr[irow, icol].scatter(
			locs_true[:, 1] * scale,
			locs_true[:, 0] * scale,
			color='blue',
			marker='o',
			label='Ground truth'
		)
		# Plot estimator center
		axarr[irow, icol].scatter(
			all_ests[i][:, 1] * scale,
			all_ests[i][:, 0] * scale,
			color='red',
			marker='x',
			label='Estimated'
		)

		# Plot rectangles
		for j in range(all_ests[i].shape[0]):
			radius = all_errors[i][j]
			if radius > 0:
				xj = scale * (all_ests[i][j, 1] - radius)
				yj = scale * (all_ests[i][j, 0] - radius)
				circle = patches.Circle(
					(xj, yj),
					scale * radius, 
					edgecolor='r',
					facecolor='r', 
					alpha=0.5
				)
				axarr[irow, icol].add_patch(circle)
				axarr[irow, icol].text(xj, yj, np.around(all_peps[i][j], 2), color='white')
				#axarr[irow, icol].text(xj, yj + 2*scale*radius, np.around(1 / radius, 4), color='white')
			
		axarr[irow, icol].set(title=est_names[i])

	plt.show()

