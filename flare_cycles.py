'''
Title: Using Flare Rates to Search for Stellar Activity Cycles
Authors: Matthew Scoggins, James Davenport, Kevin Covey
Arxiv:

Code name: flare_cycles.py

Language: Python 3.6

Description of input data: Users should have a .txt file containing all of the KICs they'd like to evaluate, one KIC per line. This is the targets.txt file in main
                           Additionally, users should have the KIC .flare files for each of the KIC's they'd like to evaluate. These .flare files were generated using
                           Dr. Davenport's 'appaloosa' available at https://github.com/jradavenport/appaloosa. These files contain a whole host of information, but we only
                           used the 'Equiv_Dur' column, which is the equivalent duration for each flare. The KICs I've used are available avaible in the KICs directory at
                           https://github.com/mscoggs/flare_cycles

Description of output data: This program outputs graphs and the fit_data associated with the fits on each of those graphs. See https://github.com/mscoggs/flare_cycles for examples

From the command line: python flare_cycles.py
'''


import emcee
from glob import glob
from pylab import *
from scipy.optimize import curve_fit
import scipy.optimize as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats



set_matplotlib_formats('pdf')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 14,10
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['font.size'] = 25
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 4
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'


#defining all of our constants
KIC_008507979_LOG_LUM = 31.14552765
BJD_2008 = 2454466.500000
EPOINT = 0
BIC_THRESHOLD = 2
BASE= 2.5
MAX_DEGREE = 1
DAY_RANGE = [0,400,800,1200,1600]
GROUPING_SIZE = [1]
FIXED_ENEGRY_LIST = []
CMAP = plt.cm.coolwarm
COLOR = plt.cm.ScalarMappable(cmap=CMAP)
COLOR.set_array(DAY_RANGE)
NAMES=("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1", "t_FWHM_aflare1", "amplitude_aflare1", "flare_chisq", "KS_d_model", "KS_p_model", "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")
EVF_DIR = 'energy_vs_frequency_plot'
EVF_SUB_DIR = 'energy_vs_frequency_subtraction_plot'
EVF_SUB_MEAN_DIR = 'energy_vs_frequency_subtraction_mean_plot'
TVF_DIR = 'time_vs_frequency_plot'
FIT_DATA_DIR = 'fit_data'
PROMISING_DIR = 'promising_bin'
NEUTRAL_DIR = 'neutral_bin'
NOISE_DIR = 'noise_bin'


#Control plots get shown with SHOWX, which error bars end up on their plots with ERRORx and saving with SAVEX
PLOT = True
SHOWE = True
SHOWES = False
SHOWM = False
SHOWT = False
ERRORE = False
ERRORES = False
ERRORM = False
ERRORT = False
SAVEPLOT = True
SAVETXT = False
OK68_CUTOFF = False





def get_size(file):
    '''
    Returning the number of KICs in a targettext file.

    Parameters:
    -------------------------------------------------------------------------
    file (string): The path to the file containing the list of KICs.


    Returns:
    -------------------------------------------------------------------------
    get_size_col.size (integer): The number of KICs in the file.
    '''

    temp = pd.read_table(file, names=['kic'])
    get_size_col = temp['kic'].values
    return get_size_col.size





def calc_chi_sq(data, fit, err_array):
    '''
    Caclulate the Chi-Squared value between a fit and the data.

    Parameters:
    -------------------------------------------------------------------------
    data (array, floats): the x-axis
    fit (array, floats): An array of the fit data for all of the data points, yaxis
    err_array (array, floats): An array of the error in each fit


    Returns:
    -------------------------------------------------------------------------
    chi_sq (float): the calculated Chi-Square Value
    '''

    chi_sq = np.sum(((data - fit) / err_array)**2.0)/np.size(data)
    chi_sq = np.sum(((data - fit))**2.0)/np.size(data)
    return chi_sq





def calc_error_xi(data, multiple):
    '''
    Calculate the asymmetric Poisson error, using Eqn 7 and Eqn 12 in Gehrels 1986 ApJ, 3030, 336. (S=1, err = mean-data)
    http://adsabs.harvard.edu/cgi-bin/nph-data_query?bibcode=1986ApJ...303..336G&link_type=ARTICLE&db_key=AST&high=

    Parameters:
    -------------------------------------------------------------------------
    data (array, floats): the x-axis
    multiple (into): The total duration (in days) of each quarter, so that we can get turn the data
                        from a #/day unit to a total #, allowing for error bar calculation

    Returns:
    -------------------------------------------------------------------------
    err_up (array, floats): the upper error bar
    err_dn (array, floats): the down error bar
    '''

    frequency = (10**data)
    count = frequency*multiple
    err_dn_count = (np.abs(count * (1.-1./(9. * count)-1./(3.*np.sqrt(count)))**3.-count))
    err_up_count = (np.sqrt(count + 0.75) + 1.0)
    err_dn_frequecny = err_dn_count/multiple
    err_up_frequecny = err_up_count/multiple
    err_dn_log_freq = 0.434*(err_dn_frequecny/frequency)
    err_up_log_freq = 0.434*(err_up_frequecny/frequency)

    return err_up_log_freq, err_dn_log_freq





def calc_bic(size, degree, chi_sq):
    '''
    Using the Bayesian Information Criterion calculation, preventing overfitting of the data.

    Parameters:
    -------------------------------------------------------------------------
    size (int): number of data points
    degree (int): degree of the polynomial fit.
    chi_sq (float): chi-squared value.

    Returns:
    -------------------------------------------------------------------------
    bic (float): the Bayesian Information Criterion Value
    '''

    bic = size*np.log(chi_sq) + ((degree+1) * np.log(size))
    return bic






def fit_degree_k(xaxis, yaxis, err_array, degree, size, fit_linspace):
    '''
    Fitting a polynomial of degree K.

    Parameters:
    -------------------------------------------------------------------------
    xaxis, yaxis (array, floats): x and y axis values that will be fit
    err_array (array, floats):  data's associated error bars
    degree (int): degree of the polynomial to be fit
    size (int): number of data points
    fit_linspace (array, floats): a dense x-axis linspace so the fit line is continuous

    Returns:
    -------------------------------------------------------------------------
    bic (float): Bayesian Information Criterion value
    fit (array, floats): the best fit of degree k
    parameters (array, floats): coefficients for the fit
    covariance (2d-array, floats): covariance matrix associated w/ the fit.
    chi_sq (float): chi-squared value
    '''

    if(degree == 0):
        parameters = [np.mean(yaxis)]
        covariance = np.array([[np.std(yaxis)**2,0],[0,0]])
    else:
        parameters, covariance = np.polyfit(xaxis, yaxis, degree, cov=True, full =False, w=(1/err_array))

    fit = np.polyval(parameters, fit_linspace)
    fit_at_data = np.polyval(parameters, xaxis)
    chi_sq = calc_chi_sq(yaxis, fit_at_data, err_array)
    bic = calc_bic(size, degree, chi_sq)
    return bic, fit, parameters, covariance, chi_sq





def append_array(fit_data, target_index, KIC, size, group_size, degree, best_parameters, best_chi_sq, best_cov, bics):
    '''
    Updating an array that contain all of the fit data.

    Parameters:
    -------------------------------------------------------------------------
    fit_data (2d-array, floats/ints): array being updated with the fit data, which will eventually be exported (written) to a file
    target_index (int): index that keeps track of which KIC we're on
    KIC (string): KIC who's data is getting sent to the array
    size (int): number of data points
    degree (int): degree of the fit
    best_parameters (array, floats): coefficients of the best fit polynomial
    best_chi_sq (float): chi-square of the best fit
    best_cov (2d-array): covariance matrix of the best-fit, returned by polyfit
    bics (4 floats): the bics from each degree of fit, 0 if there wasn't a fit

    Returns:
    -------------------------------------------------------------------------
    fit_data (2d-array, floats/ints): array being updated with the fit data, which will eventually be exported (written) to a file
    '''

    fit_data[target_index, 0] = KIC
    fit_data[target_index, 1] = size
    fit_data[target_index, 2] = group_size
    fit_data[target_index, 3] = degree
    fit_data[target_index, 4] = '{}'.format('%.5f'%best_chi_sq)
    length = len(best_parameters)

    for x in range(length):

        #working backwards because best_parameters varies in length
        fit_data[target_index, (9-length+x)] = '{}'.format('%.15f'%best_parameters[x])
        fit_data[target_index, (13-length+x)] = '{}'.format('%.15f'%(np.sqrt(best_cov[x,x])))

    for x in range(MAX_DEGREE):

        fit_data[target_index, (13+x)] = '{}'.format('%.15f'%(bics[-(x+1)]))

    return fit_data





def compare_fits(xaxis, yaxis, err_array, fit_linspace):
    '''
    Comparing two fits, and determining which fit is the 'best' based on the Bayseian information criterion

    Parameters:
    -------------------------------------------------------------------------
    xaxis, yaxis (array, floats): x and y axis values that will be fit
    err_array (array, floats):  data's associated error bars
    fit_linspace (array, floats): a dense x-axis linspace so the fit line is continuous

    Returns:
    -------------------------------------------------------------------------
    best_fit (array, floats): data from the best fit polynomial
    best_parameters (array, floats): coefficients of polynomial being used to fit the data
    best_covariance (2d-array, floats): covariance matrix of the best_fit, returned by polyfit
    best_chi_sq (array, floats): Chi-Square of the best fit
    degree_of_best_fit (int): degree of the polynomial being used to fit the data
    size (int): number of data points in the fit
    bics (array, floats): The bics from each fit. 0 if no fit was found
    '''

    bics = np.zeros(MAX_DEGREE)
    size = np.size(xaxis)
    degree_of_best_fit = 0
    bic_min, best_fit, best_parameters, best_covariance, best_chi_sq = fit_degree_k(xaxis, yaxis, err_array, degree_of_best_fit, size, fit_linspace)
    bics[0] = bic_min
    degree = 1

    while(degree<MAX_DEGREE):

        if(size <= degree+3): break

        bic_new, fit, parameters, covariance, chi_sq = fit_degree_k(xaxis, yaxis, err_array, degree, size, fit_linspace)
        bics[degree] = bic_new

        if ((bic_min - bic_new) >=BIC_THRESHOLD):

            bic_min, best_fit, best_parameters, best_covariance, best_chi_sq = bic_new, fit, parameters, covariance, chi_sq
            degree_of_best_fit = degree

        degree+=1

    return best_fit, best_parameters, best_covariance, best_chi_sq, degree_of_best_fit, size, bics





def init_data_array(num_rows):
    '''
    initializing the array that will hold the fit data, which gets exported to a .txt file

    Parameters:
    -------------------------------------------------------------------------
    num_rows (int): number of rows that we need in the fit_data.txt

    Returns:
    -------------------------------------------------------------------------
    fit_data (2d-array, floats/ints): array being updated with the fit data, which will eventually be exported (written) to a file
    target_index (int): keeps track of which row we're on
    '''

    fit_data = np.zeros((num_rows, 17), dtype='O')
    fit_data[0] = ["#This is a file containing the data involved in the best fit of our KICs",'','','','','','','','','','','','','','','','']
    fit_data[1] = ["#KIC", 'N', 'group size', 'best degree', 'chi_sq', 'X^3', 'X^2', 'X^1', 'X^0', 'Error3', 'Error2', 'Error1', 'Error0', 'BIC3', 'BIC2','BIC1','BIC0']
    target_index = 2
    return fit_data, target_index





def get_label(best_parameters, degree_of_best_fit, group_size):
    '''
    Creates the label that will show up on graphs

    Parameters:
    -------------------------------------------------------------------------
    best_parameters (array, floats): coefficients of polynomial being used to fit the data
    degree_of_best_fit (int): degree of the polynomial being used to fit the data
    group_size (int): number of points that go into each grouping

    Returns:
    -------------------------------------------------------------------------
    label (string): a graph-ready label
    '''

    label = "BIC-fit\nP_0: "+ str('%.2E' % Decimal(best_parameters[-1]))

    #adding the parameters to the label
    for x in range(degree_of_best_fit): label = label + "\nP_"+str(x+1)+": "+str('%.2E' % Decimal(best_parameters[-(x+2)]))
    label = label + "\ngroup size: " + str(group_size)
    return label





def calc_error_during_subtraction(data, data_err, fit_coeff, coeff_err, total_quarter_duration):
    '''
    Propogates the error during the subtraction step.

    Parameters:
    -------------------------------------------------------------------------
    data (array, floats): yaxis, frequency
    data_err (array, floats): error in the yaxis, poisson error
    fit_coeff (array, floats): coefficients involve in the fit done by curve_fit
    coeff_err (array, floats): error in the coefficients
    total_quarter_duration (int): total number of days that the star was observed

    Returns:
    -------------------------------------------------------------------------
    difference_err (array, floats): the propogated errors
    '''
    a,b = fit_coeff[0],fit_coeff[1]
    da,db = coeff_err[0],coeff_err[1]
    #a,b,c = fit_coeff[0],fit_coeff[1],fit_coeff[2]
    #da,db,dc = coeff_err[0],coeff_err[1],coeff_err[2]

    #using df = sqrt((df/dx * dx)^2  +  (df/dy * dy)^2 + ...) where f is our powerlaw
    # f = a*base**(-bx)
    # df/da = base**(-bx)
    # df/db = a*base**(-bx) * -x*ln(base)
    df_squared = (a * (BASE**(-b*data)) * data * log(BASE) * db)**2    +    ((BASE**(-b*data))*da)**2

    #if f = A + B, df = sqrt(da^2 + db^2)
    difference_err = np.sqrt(df_squared + data_err**2)
    return difference_err





def power_law(x, a, b):
    '''A powerlaw function, parameters determined by curve_fit'''
    return a*x+b





def lnlike(theta, x, y, yerr):
    ''' an mcmc auxiliary function'''
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))





def lnprior(theta):
    ''' an mcmc auxiliary function'''
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf





def lnprob(theta, x, y, yerr):
    ''' an mcmc auxiliary function'''
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)





def mcmc(x,y,yerr, m_guess, b_guess):
    '''
    Doing an mcmc fit on x and y

    Parameters:
    -------------------------------------------------------------------------
    x (array): x-axis data
    y (array): x-axis data
    yerr (array): yerr-axis data
    m_guess (float): a close estimate to the slope, generated by curve_fit
    b_guess (float): a close estimate to the y-offset, generated by curve_fit

    Returns:
    -------------------------------------------------------------------------
    [ m_mcmc[0], b_mcmc[0]], [ m_mcmc[1],  b_mcmc[1]] : the mcmc [slope, y-offset], [slope-error, y-error]
    '''
    f_true = 0.5
    N = np.size(x)
    yerr = 0.1+0.5*np.random.rand(N)
    nll = lambda *args: -lnlike(*args)

    result = op.minimize(nll, [m_guess, b_guess, np.log(f_true)], args=(x, y, yerr))
    m_ml, b_ml, lnf_ml = result["x"]
    ndim, nwalkers = 3, 100
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    sampler.run_mcmc(pos, 500)

    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    xl = np.array([np.min(x), np.max(x)])

    for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(x+lum, m*x+b, color="k", alpha=0.1)
    samples[:, 2] = np.exp(samples[:, 2])
    m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0)))
    plt.plot(x+lum, m_mcmc[0]*x+ b_mcmc[0], color="r", lw=2, alpha=0.8)
    #print("b: ",b_mcmc, "m: ",m_mcmc)
    return [ m_mcmc[0], b_mcmc[0]], [ m_mcmc[1],  b_mcmc[1]]





def plot_tvf(KIC, files, num_files, tvf_data, fixed_energy, target_index, **kwargs):
    '''
    plotting the time vs frequency for each KIC, at a fixed energy. Choose a fixed energy, and get the frequency
    for each quarter at the energy

    Parameters:
    -------------------------------------------------------------------------
    KIC (string): name of the star being studied
    files: All of the quarters, .flare files
    num_files (int): total number of quarters
    tvf_data (2d-array, floats/int): the array that will become the file holding all of the fit data for tvf fit
    fixed_energy (float): energy the we're fixing on the xaxis of evf
    target_index (int): an index that keeps track of which row we're on in tvf_data
    '''

    if(PLOT):

        plt.figure()
        plt.ylabel(r"$\nu$")
        plt.xlabel("$BJD_{TDB}-2454832$")
        plt.yscale('log')

    time = np.array([])
    frequency = np.array([])
    err_array_up = np.array([])
    err_array_dn = np.array([])

    for x in range(num_files):

        total_quarter_duration = pd.read_table(files[x], skiprows=5, nrows=1, header=None, delim_whitespace=True, usecols=(7,)).iloc[0].values[0]
        df = pd.read_table(files[x], comment="#", delimiter=",", names=NAMES)

        energy = np.array(df['Equiv_Dur'])
        positive = np.where(energy > 0)
        energy_p = energy[positive]
        sort = np.argsort(energy_p)
        evf_x_energy = (np.log10(energy_p) + EPOINT)[sort][::-1]
        evf_x_energy = evf_x_energy[np.isfinite(evf_x_energy)]
        evf_y_frequency = (np.arange(1, len(evf_x_energy)+1, 1))/total_quarter_duration

        if(len(evf_x_energy) == 0): continue

        if(np.amax(evf_x_energy) >= fixed_energy >= np.amin(evf_x_energy)):

            mean_start = np.sum(df['t_start'])/len(df['t_start'])

            ffdYAtEnergy = np.interp(fixed_energy, evf_x_energy[::-1], evf_y_frequency[::-1])
            time = np.append(time, mean_start)
            frequency = np.append(frequency, ffdYAtEnergy)
            err_up, err_dn = calc_error_xi(ffdYAtEnergy, total_quarter_duration)
            err_array_up = np.append(err_array_up, err_up)
            err_array_dn = np.append(err_array_dn, err_dn)

    if(len(time) == 0): return

    fit_linspace = np.linspace(min(time), max(time), num=100)
    best_fit, best_parameters, best_covariance, best_chi_sq, degree_of_best_fit, size, bics = compare_fits(time, frequency, err_array_up, fit_linspace)
    tvf_data = append_array(tvf_data, target_index, KIC, size, 1, degree_of_best_fit, best_parameters, best_chi_sq, best_covariance, bics)

    label = get_label(best_parameters, degree_of_best_fit, 1)

    if(PLOT):
        plt.scatter(time, frequency, c=time, cmap=CMAP, s=400, edgecolors = 'black', alpha = 1)
        plt.plot(fit_linspace, best_fit, 'black', lw=4, label=label)
        if(kwargs['errort']==True): plt.errorbar(time, frequency, yerr = [err_array_dn,err_array_up], c='black', fmt='o', markersize=0, elinewidth=.8, capsize=6)#plotting error
        if(kwargs['save']==True): plt.savefig('plots/tvf/'+ str(KIC) + '_tvf.pdf')
        if(kwargs['showt']==True): plt.show()
        plt.close()





def plot_fractional_lum(KIC, files, num_files, **kwargs):
    '''
    plotting the fractional luminosity, L_flares / L_star  = sum(equivalent duration /exposure time) over each quarter, looking for coherent changes
    over time which might indicate a stellar activity cycle

    Parameters:
    -------------------------------------------------------------------------
    KIC (string): name of the star being studied
    files: All of the quarters, .flare files
    num_files (int): total number of quarters
    tvf_data (2d-array, floats/int): the array that will become the file holding all of the fit data for tvf fit
    fixed_energy (float): energy the we're fixing on the xaxis of evf
    target_index (int): an index that keeps track of which row we're on in tvf_data
    '''


    if(PLOT):
        plt.figure()
        plt.ylabel(r'log( $L_{fl}$ / $L_{Kp}$ )')
        plt.xlabel("$BJD_{TDB}-2454832$")

    time = np.array([])
    log_fractional_lum  = np.array([])

    for x in range(num_files):

        total_quarter_duration = pd.read_table(files[x], skiprows=5, nrows=1, header=None, delim_whitespace=True, usecols=(7,)).iloc[0].values[0]
        df = pd.read_table(files[x], comment="#", delimiter=",", names=NAMES)

        #grabbing the energy (equivalent duration) column from each file, sorting, then including only the positive values so it can be logged
        energy = np.array(df['Equiv_Dur'])
        positive = np.where(energy > 0)
        energy_p = energy[positive]

        time = np.append(time, np.sum(df['t_start'])/len(df['t_start']))
        log_fractional_lum = np.append(log_fractional_lum, np.log10(np.sum(energy_p)/total_quarter_duration))




    try:
        err_array = np.zeros(np.size(time))
        popt, pcov = curve_fit(power_law, time, log_fractional_lum, p0=(-.5, .4),maxfev = 100000)
        perr = np.sqrt(np.diag(pcov))
        popt, perr = mcmc(time, log_fractional_lum, err_array, popt[0], popt[1])
        plt.scatter(time, log_fractional_lum, cmap=CMAP, s=400, c = time, edgecolors = 'black', alpha =1)

        if(PLOT):
            if(kwargs['save']==True): plt.savefig('plots/fractional_lum/'+ str(KIC) + '_frac_lum.pdf')
            if(kwargs['showe']==True): plt.show()
            plt.close()

    except Exception as e:
        print("ERROR: Couldn't do fractional_lum analysis to " + KIC + ". Moving on to the next KIC")
        print("EXCEPTION: ",e)





def plot_evf(KIC, files, num_files, **kwargs):
    '''
    Plotting a reverse cummulative sum of the flare frequency for each quarter.

    Parameters:
    -------------------------------------------------------------------------
    KIC (string): name of the star being studied
    files: All of the quarters, .flare files
    num_files (int): total number of quarters

    Returns:
    -------------------------------------------------------------------------
    quarterly_evf_x_energy (2d-array, floats): each element is an array holding the x axis, energy, for a quarter
    quarterly_evf_y_frequency (2d-array, floats): each element is an array holding the y axis, frequency, for a quarter
    popt (array, floats): the curve_fit parameters
    perr (array, floats): the error in the curve_fit parameters
    total_quarter_duration (float): the total number of days for a KIC
    time (array, floats): the mean times for each quarter
    success (bool): curve_fit rarely fails to find a fit. If it fails, we can't do the subtraction and mean analysis
    '''


    if(PLOT):
        f =plt.figure()
        plt.ylabel("log( " + r"$\nu$ )")
        if(KIC == "008507979"): plt.xlabel("log( Flare Energy ) (erg)")
        else: plt.xlabel("log( Equivalent Duration )")

    err_array = np.array([])
    total_evf_x_energy = np.array([])
    total_log_evf_y_frequency = np.array([])
    quarterly_evf_x_energy = []
    quarterly_log_evf_y_frequency = []
    time = np.array([])
    flare_total = 0
    duration_total = 0


    for x in range(num_files):


        total_quarter_duration = pd.read_table(files[x], skiprows=5, nrows=1, header=None, delim_whitespace=True, usecols=(7,)).iloc[0].values[0]
        df = pd.read_table(files[x], comment="#", delimiter=",", names=NAMES)


        energy = np.array(df['Equiv_Dur'])
        positive = np.where(energy > 0)
        energy_p = energy[positive]
        sort = np.argsort(energy_p)
        evf_x_energy = np.log10((energy_p + EPOINT)[sort][::-1])
        log_evf_y_frequency = np.log10((np.arange(1, len(evf_x_energy)+1, 1))/total_quarter_duration)


        if(kwargs['OK68']==True): ok68 = (evf_x_energy >= np.log10(np.median(df['ED68i'])) + EPOINT)
        else: ok68 = np.isfinite(evf_x_energy)


        flare_total += len(evf_x_energy[np.where((evf_x_energy+KIC_008507979_LOG_LUM)>32)])
        duration_total += total_quarter_duration


        if (any(ok68)):
            quarterly_evf_x_energy.append(evf_x_energy[ok68])
            quarterly_log_evf_y_frequency.append(log_evf_y_frequency[ok68])
            time = np.append(time, np.sum(df['t_start'])/len(df['t_start']))


        total_evf_x_energy = np.append(total_evf_x_energy, evf_x_energy[ok68])
        total_log_evf_y_frequency = np.append(total_log_evf_y_frequency, log_evf_y_frequency[ok68])


        err_up, err_dn = calc_error_xi(log_evf_y_frequency[ok68], total_quarter_duration)
        err_array = np.append(err_array, err_up)


        if(PLOT):
            #converting from equiv dur to luminosity for the KIC being used in the paper
            if(KIC== "008507979"): plt.plot(evf_x_energy[ok68]+KIC_008507979_LOG_LUM, log_evf_y_frequency[ok68], lw = 3, c = CMAP(x/float(len(files))))
            else: plt.plot(evf_x_energy[ok68]+KIC_008507979_LOG_LUM, log_evf_y_frequency[ok68], lw = 3, c = CMAP(x/float(len(files))))
            if(kwargs['errore']==True): plt.errorbar(evf_x_energy[ok68], log_evf_y_frequency[ok68], yerr = [err_dn, err_up], c = 'black', elinewidth=.3, fmt='o', markersize = .55)

    flares_per_day = str(flare_total/duration_total)[0:4]
    print("Flares per day: ", flares_per_day)
    sort = np.argsort(total_evf_x_energy)
    if(len(total_evf_x_energy) == 0):  return 0, 0, 0, 0, 0, 0, False


    try: #sometimes curve_fit throws an error becuase it can't find a fit after a certain number of tries. If it can't, just move on to the next KIC
        popt, pcov = curve_fit(power_law, total_evf_x_energy[sort], total_log_evf_y_frequency[sort], p0=(-.5, .4),maxfev = 100000)
        perr = np.sqrt(np.diag(pcov))

        if(PLOT):
            if(kwargs['save']==True): plt.savefig('plots/evf/'+ str(KIC) + '_evf.pdf')
            if(kwargs['showe']==True): plt.show()
            plt.close()

        success = True

    except Exception as e:
        print("ERROR: Couldn't fit a power_law to " + KIC + ". Moving on to the next KIC")
        print("EXCEPTION: ",e)
        success = False
        popt = perr = 0

    return quarterly_evf_x_energy, quarterly_log_evf_y_frequency, popt, perr, total_quarter_duration, time, success





def plot_evf_sub(KIC, quarterly_evf_energy, quarterly_evf_log_frequency, popt,perr, total_quarter_duration, **kwargs):
    '''
    Takes the fit from plot_evf, and each of the quarters, and finds the difference between them.

    Parameters:
    -------------------------------------------------------------------------
    KIC (string): name of the star being studied
    quarterly_evf_x_energy (2d-array, floats): each element is an array holding the x axis, energy, for a quarter
    quarterly_evf_log_frequency (2d-array, floats): each element is an array holding the y axis, frequency, for a quarter
    popt (array, floats): the curve_fit parameters
    perr (array, floats): the error in the curve_fit parameters
    total_quarter_duration (float): the total number of days for a KIC

    Returns:
    -------------------------------------------------------------------------
    mean_frequency (array, floats): the mean frequency for each quarter's difference
    mean_frequency_err (array, floats): the error for the mean frequency for each quarter's difference
    '''

    if(PLOT):

        plt.figure()
        #plt.ylabel(r'$\nu$ - $\bar \nu$')
        plt.ylabel(r'log  $\nu$  - fit')
        plt.xlabel("log( Equivalent Duration )")

    mean_log_frequency = np.array([])
    mean_log_frequency_err = np.array([])
    error_per_quarter = np.array([])

    for q in range(len(quarterly_evf_energy)):

        fit = power_law(quarterly_evf_energy[q], *popt)
        err_up, err_dn = calc_error_xi(quarterly_evf_log_frequency[q], total_quarter_duration)
        error_per_quarter = np.append(error_per_quarter, np.sqrt(np.sum(err_up**2))/np.size(err_up))

        difference_err = err_up #calc_error_during_subtraction(quarterly_evf_x_energy[q]t, err_up, popt, perr ,total_quarter_duration)
        difference = quarterly_evf_log_frequency[q]-fit

        if(PLOT): plt.plot(quarterly_evf_energy[q], difference, c = CMAP(q/float(len(quarterly_evf_energy))))
        if(kwargs['errores']==True): plt.errorbar(quarterly_evf_energy[q], difference, yerr = [difference_err, difference_err], c = 'black', elinewidth=.3, fmt='o', markersize = .55)


        mean_log_per_quarter = np.sum(difference / (difference_err**2)) / np.sum(1/(difference_err**2))

        mean_log_frequency = np.append(mean_log_frequency, mean_log_per_quarter)
        mean_log_frequency_err = np.append(mean_log_frequency_err, np.sqrt(1/(np.sum(1/(difference_err**2)))))

    if(PLOT):
        if(kwargs['save']==True):plt.savefig('plots/evf_sub/'+ str(KIC) + '_evf_sub.pdf')
        if(kwargs['showes']==True): plt.show()
        plt.close()

    return mean_log_frequency, mean_log_frequency_err, error_per_quarter





def plot_evf_sub_mean(KIC, time, mean_frequency, mean_frequency_err,  group_size, evf_sub_mean_data,error_per_quarter, target_index,**kwargs):
    '''
    plots the mean difference for each quarter (calculated in plot_evf_sub) over time

    Parameters:
    -------------------------------------------------------------------------
    KIC (string): name of the star being studied
    time (array, floats): the mean times for each quarter
    mean_frequency (array, floats): the mean frequency for each quarter's difference
    mean_frequency_err (array, floats): the error for the mean frequency for each quarter's difference
    group_size (int): number of points that go into each grouping
    evf_sub_mean_data (2d-array, floats/int): the array that will become the file holding all of the fit data for efv_sub_mean fit
    target_index (int): an index that keeps track of which row we're on in tvf_data
    '''
    mean_frequency_err = error_per_quarter
    if(group_size == 1): grouped_time, grouped_mean_vals, grouped_mean_errs = time, mean_frequency, mean_frequency_err
    if(group_size == 1): grouped_time, grouped_mean_vals, grouped_mean_errs = time, mean_frequency, error_per_quarter


    else:

        grouped_time = np.zeros([math.ceil(len(time)/group_size)])
        grouped_mean_vals = np.zeros([math.ceil(len(time)/group_size)])
        grouped_mean_errs = np.zeros([math.ceil(len(time)/group_size)])
        index = iterations = 0

        for x in range(len(time)):

            grouped_time[index] += time[x]
            grouped_mean_vals[index] += mean_frequency[x]
            grouped_mean_errs[index] += mean_frequency_err[x]**2
            iterations += 1

            if((iterations == group_size) or (x == (len(time)-1))):

                grouped_time[index] = grouped_time[index]/iterations
                grouped_mean_vals[index] = grouped_mean_vals[index]/iterations
                grouped_mean_errs[index] = np.sqrt(grouped_mean_errs[index])/iterations
                iterations = 0
                index += 1

    plt.figure()
    plt.ylabel(r'$\overline{log(\nu) - fit}$')
    plt.xlabel("$BJD_{TDB}-2454832$")

    #creating a dense linespace so the fit is smooth
    fit_linspace = np.linspace(min(grouped_time), max(grouped_time), num=100)
    best_fit, best_parameters, best_covariance, best_chi_sq, degree_of_best_fit, size, bics = compare_fits(grouped_time, grouped_mean_vals, grouped_mean_errs, fit_linspace)

    popt, perr = mcmc(grouped_time,grouped_mean_vals,grouped_mean_errs, best_parameters[0], best_parameters[0])
    best_fit = popt[0]*fit_linspace + popt[1]
    best_parameters = popt
    evf_sub_mean_data = append_array(evf_sub_mean_data, target_index, KIC, size, group_size, degree_of_best_fit, best_parameters, best_chi_sq, best_covariance, bics)

    if(PLOT):
        plt.scatter(grouped_time, grouped_mean_vals, c=grouped_time, cmap=CMAP, s=400, edgecolors = 'black', alpha=1)
        if(kwargs['errorm']==True): plt.errorbar(grouped_time, grouped_mean_vals, yerr = grouped_mean_errs, c='black', fmt='o', markersize=0, elinewidth=.8, capsize=6)
        if(kwargs['save']==True): plt.savefig('plots/evf_sub_mean/'+ str(KIC) + '_evf_sub_mean_groupsize_' + str(group_size) + '.pdf')
        if(kwargs['showm']==True): plt.show()
    plt.close()





def main():
        file = 'target_single.txt'
        target_count = get_size(file)
        targets = open(file, "r")

        evf_sub_mean_data, target_index = init_data_array(target_count*len(GROUPING_SIZE) + 2)

        for line in targets:

            KIC = line.rstrip('\n')
            files = sorted(glob('KICs/'+KIC+"/*.flare"))
            num_files = len(files)
            print("\nWorking on the fractional_lum analysis for KIC: "+str(KIC)+",   " + str(target_index-1) + "/" +str(target_count))
            fractional_lum(KIC, files, num_files, showe=SHOWE,errore=ERRORE,save=SAVEPLOT, OK68=OK68_CUTOFF)

            print("Working on the energy_vs_frequency analysis")
            quarterly_evf_energy, quarterly_evf_log_frequency, popt, perr, total_quarter_duration, time, success  = plot_evf(KIC, files, num_files, showe=SHOWE,errore=ERRORE,save=SAVEPLOT, OK68=OK68_CUTOFF)

            if(success):
                print("Working on the plot_evf_sub analysis")
                mean_log_frequency, mean_log_frequency_err, error_per_quarter = plot_evf_sub(KIC, quarterly_evf_energy, quarterly_evf_log_frequency, popt,perr, total_quarter_duration,showes=SHOWES,errores=ERRORES, save=SAVEPLOT)

                for group_size in GROUPING_SIZE:
                    print("Working on the plot_evf_sub_mean analysis for group_size = ", group_size)
                    plot_evf_sub_mean(KIC, time, mean_log_frequency, mean_log_frequency_err, group_size, evf_sub_mean_data,error_per_quarter, target_index,errorm=ERRORM,showm=SHOWM,save=SAVEPLOT)
                    target_index+= 1

        targets.close()
        #if(SAVETXT==True): np.savetxt(bin_+'/'+FIT_DATA_DIR+'/evf_mean_sub.txt', evf_sub_mean_data, fmt = '% 20s', delimiter=' ', newline='\n', header='', footer='', comments='# ')


        for energyConstant in FIXED_ENEGRY_LIST:

            targets = open(file, "r")
            fixed_energy = energyConstant + EPOINT
            tvf_data, target_index = init_data_array(target_count + 2)

            for line in targets:

                KIC = line.rstrip('\n')
                print("Working on the time_vs_frequency analysis for KIC: "+str(KIC)+ " at energy: "+str(fixed_energy))
                files = glob('KICs/'+KIC+"/*.flare")
                num_files = len(files)
                plot_tvf(KIC, files, num_files, tvf_data, fixed_energy, target_index, showt=SHOWT, errort = ERRORT,save=SAVEPLOT)
                target_index += 1

            targets.close()
            #if(SAVETXT==True): np.savetxt(bin_+'/'+FIT_DATA_DIR+'/fixed_energy_equals_'+str(fixed_energy)+'.txt', tvf_data, fmt = '% 20s', delimiter=' ', newline='\n', header='', footer='', comments='# ')

main()
