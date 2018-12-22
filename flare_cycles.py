import sys
import os
import shutil
from glob import glob
from decimal import Decimal
from pylab import *
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 14,10
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['font.size'] = 25
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 4
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
BJD_2008 = 2454466.500000
EPOINT = 0
bic_threshold = 2
base= 2.5
cmap = plt.cm.coolwarm
color = plt.cm.ScalarMappable(cmap=cmap)
day_range = [0,400,800,1200,1600]
color.set_array(day_range)
names=("t_start", "t_stop", "t_peak", "amplitude", "FWHM", "duration", "t_peak_aflare1",
       "t_FWHM_aflare1", "amplitude_aflare1", "flare_chisq", "KS_d_model", "KS_p_model",
       "KS_d_cont", "KS_p_cont", "Equiv_Dur", "ED68i", "ED90i")



evf_dir = 'energy_vs_frequency_plot'
evf_sub_dir = 'energy_vs_frequency_subtraction_plot'
evf_sub_mean_dir = 'energy_vs_frequency_subtraction_mean_plot'
tvf_dir = 'time_vs_frequency_plot'
fit_data_dir = 'fit_data'
promising_dir = 'promising_bin'
neutral_dir = 'neutral_bin'
noise_dir = 'noise_bin'
bin_list = [promising_dir, neutral_dir, noise_dir]

for bin_ in bin_list:
    if not os.path.exists(bin_): #searching for, and making the directories if they don't exist
        os.makedirs(bin_)
    if not os.path.exists(bin_+'/'+evf_dir): #searching for, and making the directories if they don't exist
        os.makedirs(bin_+'/'+evf_dir)
    if not os.path.exists(bin_+'/'+evf_sub_dir):
        os.makedirs(bin_+'/'+evf_sub_dir)
    if not os.path.exists(bin_+'/'+evf_sub_mean_dir):
        os.makedirs(bin_+'/'+evf_sub_mean_dir)
    if not os.path.exists(bin_+'/'+tvf_dir):
        os.makedirs(bin_+'/'+tvf_dir)
    if not os.path.exists(bin_+'/'+fit_data_dir):
        os.makedirs(bin_+'/'+fit_data_dir)

PLOT = True
SHOWE = False
SHOWES = False
SHOWM = False
SHOWT = False
ERRORE = False
ERRORM = True
ERRORT = True
SAVEPLOT = True
SAVETXT = True
WHOLE = False




##############################################################################################################################################################
#A collection of functions
##############################################################################################################################################################
def getSize(file):
    '''Returning the number of KICs in a targettext file.

    Parameters
    ----------
    file : string
        The path to the file containing the list of KICs.


    Returns
    -------
    getsize.size : integer
        The number of KICs in the file.
    '''
    temp = pd.read_table(file, names=['kic'])
    getsize = temp['kic'].values
    return getsize.size



def calcChiSq(data, fit, errList):
    '''Caclulate the Chi-Squared value between a fit and the raw data

    Parameters
    ----------
    data : array
        An array of the flare frequency data.

    fit : array
        An array of the fit data

    errList : array
        An array of the error in each fit


    Returns
    -------
    chiSq : float
        the calculated Chi-Square Value
    '''
    chiSq = np.sum(((data - fit) / errList)**2.0)/np.size(data)
    chiSq = np.sum(((data - fit))**2.0)/np.size(data)
    return chiSq



def calcError(data, multiple):
    '''Calculating the asymmetric error bar for each data point

    Parameters
    ----------
    data : array
        An array of the flare frequency data.

    multiple : integer
        The total duration (in days) of each quarter, so that we can get turn the data
        from a #/day unit to a total #, allowing for error bar calculation


    Returns
    -------
    errUp : float
        the upper error bar

    errDn : flaot
        the lower error bar
    '''
    data = data*multiple
    errDn = (np.abs(data * (1.-1./(9. * data)-1./(3.*np.sqrt(data)))**3.-data))/multiple
    errUp = (np.sqrt(data + 0.75) + 1.0)/multiple
    return errUp, errDn





def calcBic(size, degree, chiSq):
    '''Using the Bayesian Information Criterion calculation

    Parameters
    ----------
    size : integer
        The number of data points.
    degree: integer
        The degree of the polynomial fit.
    chiSq : float
        Chi-Squared value.


    Returns
    -------
    bic : float
        the Bayesian Information Criterion Value
    '''
    bic = size*np.log(chiSq) + ((degree+1) * np.log(size))
    #aic = size*np.log(chiSq) + ((degree+1) * 2)
    return bic #aic



def fitDegreeK(xaxis, yaxis, errList, degree, size, fit_linspace):
    '''Fitting a polynomial of degree K

    Parameters
    ----------
    xaxis,yaxis : arrays
        The x and y axis values that will be fit.

    errList : array
        The data's associted error bars.

    degree : integer
        The degree of the polynomial to be fit.

    size : integer
        The number of data points.


    Returns
    -------
    bic : float
        The Bayesian Information Criterion value.

    fit : array
        An array containing the best fit of degree k.

    parameters : array
        Coefficients for the fit.

    covariance : matrix (array)
        The covariance matric associated w/ the fit.

    chiSq : float
        The Chi-Squared value.
    '''

    if(degree == 0):
        parameters = [np.mean(yaxis)]
        covariance = np.array([[np.std(yaxis)**2,0],[0,0]])
    else:
        parameters, covariance = np.polyfit(xaxis, yaxis, degree, cov=True, full =False, w=(1/errList))

    fit = np.polyval(parameters, fit_linspace)
    fit_at_data = np.polyval(parameters, xaxis)
    chiSq = calcChiSq(yaxis, fit_at_data, errList)
    bic = calcBic(size, degree, chiSq)
    return bic, fit, parameters, covariance, chiSq




def updateArray(fit_data, targetIndex, KIC, size, group_size, degree, bestParameters, bestChiSquare, bestCov, bics):
    '''Updating an array that contain all of the relevant fit data

    Parameters
    ----------
    fit_data : array
        The array that will be updated.

    targetIndex : integer
        The index that keeps track of which KIC we're on.

    KIC : string
        The KIC who's data is getting sent to the array.

    size : integer
        Number of data points.

    degree : interger
        The degree of the fit.

    bestParameters : array
        The coefficients of the best fit polynomial.

    bestChiSquare : float
        The Chi-Square of the best fit.

    bestCov : array
        The covariance matrix of the best-fit, returned by polyfit.


    Returns
    -------
    fit_data : array
        The array being updated with the fit data, which will eventually be exported (written) to a file
    '''
    fit_data[targetIndex, 0] = KIC
    fit_data[targetIndex, 1] = size
    fit_data[targetIndex, 2] = group_size
    fit_data[targetIndex, 3] = degree
    fit_data[targetIndex, 4] = '{}'.format('%.5f'%bestChiSquare)
    length = len(bestParameters)

    for x in range(length):
        fit_data[targetIndex, (9-length+x)] = '{}'.format('%.15f'%bestParameters[x])
        fit_data[targetIndex, (13-length+x)] = '{}'.format('%.15f'%(np.sqrt(bestCov[x,x])))
    for x in range(4):
        fit_data[targetIndex, (13+x)] = '{}'.format('%.15f'%(bics[-(x+1)]))

    return fit_data





def compareFits(xaxis, yaxis, errList, fit_linspace):
    '''Comparing two fits, and determining which fit is the 'best' based on the Bayseian information criterion

    Parameters
    ----------
    xaxis,yaxis : arrays
        The x and y axis values that will be fit.

    errList : array
        The error associated with each point

    Returns
    -------
    bestFit : array
        The data from the best fit polynomial.

    bestParameters : array
        The coefficients of polynomial being used to fit the data.

    bestCovariance : array
        A covariance matrix of the bestfit, returned by polyfit

    bestChiSquare : float
        The Chi-Square of the best fit

    bestFitDegree : integer
        The degree of the polynomial being used to fit the data.

    size: integer
        The number of data points in the fit
    '''
    bics = np.zeros(4)
    size = np.size(xaxis)
    bestFitDegree = 0
    bic_min, bestFit, bestParameters, bestCovariance, bestChiSquare = fitDegreeK(xaxis, yaxis, errList, bestFitDegree, size, fit_linspace)
    bics[0] = bic_min
    degree = 1

    while(degree<4):

        if(size <= degree+3): # number of data points must exceed order + 2, order = degree+1
            break
        bic_new, fit, parameters, covariance, chiSquare = fitDegreeK(xaxis, yaxis, errList, degree, size, fit_linspace)
        bics[degree] = bic_new

        '''except Exception as e:
            print("ERROR: Couldn't fit the BIC-polynomial for degree "+str(degree))
            print("EXCEPTION: ", e)
            bic_new = 999
            bics[degree] = bic_new
            break'''

        if ((bic_min - bic_new) >=bic_threshold):
            bic_min, bestFit, bestParameters, bestCovariance, bestChiSquare = bic_new, fit, parameters, covariance, chiSquare
            bestFitDegree = degree
        degree+=1

    return bestFit, bestParameters, bestCovariance, bestChiSquare, bestFitDegree, size, bics



def init_data_array(line_num):
    array = np.zeros((line_num, 17), dtype='O')
    array[0] = ["#This is a file containing the data involved in the best fit of our KICs",'','','','','','','','','','','','','','','','']
    array[1] = ["#KIC", 'N', 'group size', 'best degree', 'chiSquare', 'X^3', 'X^2', 'X^1', 'X^0', 'Error3', 'Error2', 'Error1', 'Error0', 'BIC3', 'BIC2','BIC1','BIC0']
    targetIndex = 2
    return array, targetIndex



def get_label(bestParameters, bestFitDegree, group_size):
    label = "BIC-fit\nP_0: "+ str('%.2E' % Decimal(bestParameters[-1]))
    for x in range(bestFitDegree):
        label = label + "\nP_"+str(x+1)+": "+str('%.2E' % Decimal(bestParameters[-(x+2)]))
    label = label + "\ngroup size: " + str(group_size)
    return label


def calc_error_during_subtraction(data, data_err, fit_coeff, coeff_err, toteDuration):
    a = fit_coeff[0]
    b = fit_coeff[1]
    c = fit_coeff[2]
    da = coeff_err[0]
    db = coeff_err[1]
    dc = coeff_err[2]
    df_squared = (a* data * base**(-b*data)*db)**2 + (base**(-b*data)*da)**2 + dc**2
    difference_err = np.sqrt(df_squared + data_err**2)
    return difference_err


def power_law(x, a, b, c):
    return a*base**(-b*x)+c



def plot_tvf(KIC, files, fileCount, tvf_data, fixedEnergy, targetIndex, bin_, **kwargs):
    if(PLOT):
        plt.figure()
        plt.title(str(KIC))
        plt.ylabel(r"$\nu$")
        plt.xlabel("$BJD_{TDB}-2454832$")
        plt.yscale('log')

    xaxis = np.array([])
    yaxis = np.array([])
    errListUp = np.array([])
    errListDn = np.array([])



    #loop over each .flare file
    for x in range(fileCount):

        toteDuration = pd.read_table(files[x], skiprows=5, nrows=1, header=None, delim_whitespace=True, usecols=(7,)).iloc[0].values[0] #getting the total duration of each file
        df = pd.read_table(files[x], comment="#", delimiter=",", names=names)
        energy = np.array(df['Equiv_Dur']) #This is the energy column of the flare data
        positive = np.where(energy > 0)
        energy_p = energy[positive]
        sort = np.argsort(energy_p) #get inp.wherendices that would sort the energy array
        ffdXEnergy = (np.log10(energy_p) + EPOINT)[sort][::-1] #log the reverse of sorted energy
        ffdXEnergy = ffdXEnergy[np.isfinite(ffdXEnergy)]
        ffdYFrequency = (np.arange(1, len(ffdXEnergy)+1, 1))/toteDuration #get evenly spaced intervals, divide by totedur to get flares/day

        if(len(ffdXEnergy) == 0):
            continue

        if(np.amax(ffdXEnergy) >= fixedEnergy >= np.amin(ffdXEnergy)): #checking that the energy constant isn't out of bound,otherwise, interpolate doesn't work

            meanStart = np.sum(df['t_start'])/len(df['t_start']) #finding the mean time for a file
            ffdYAtEnergy = np.interp(fixedEnergy, ffdXEnergy[::-1], ffdYFrequency[::-1])#interpolating the ffd_y
            xaxis = np.append(xaxis, meanStart) #making lists so we can fit a line later
            yaxis = np.append(yaxis, ffdYAtEnergy)

            errUp, errDn = calcError(ffdYAtEnergy, toteDuration)#dealing w/ error
            errListUp = np.append(errListUp, errUp)
            errListDn = np.append(errListDn, errDn)

    if(len(xaxis) == 0): return
    fit_linspace = np.linspace(min(xaxis), max(xaxis), num=100)
    bestFit, bestParameters, bestCovariance, bestChiSquare, bestFitDegree, size, bics = compareFits(xaxis, yaxis, errListUp, fit_linspace)
    tvf_data = updateArray(tvf_data, targetIndex, KIC, size, 1, bestFitDegree, bestParameters, bestChiSquare, bestCovariance, bics)

    label = get_label(bestParameters, bestFitDegree, 1)
    if(PLOT):
        plt.scatter(xaxis, yaxis, c=xaxis, cmap=cmap, lw=6)
        plt.plot(fit_linspace, bestFit, 'black', lw=4, label=label)
        if(kwargs['errort']==True): plt.errorbar(xaxis, yaxis, yerr = [errListDn,errListUp], c='black', fmt='o', markersize=0, elinewidth=.8, capsize=6)#plotting error
        plt.legend(loc="upper right")

        if(kwargs['save']==True): plt.savefig(bin_+'/'+tvf_dir+'/'+str(KIC)+'_energy_equals'+str(fixedEnergy)+'.png')
        if(kwargs['showt']==True): plt.show()
        plt.close()





def plot_evf(KIC, files, fileCount,bin_, **kwargs):
    if(PLOT):
        plt.figure()
        plt.title(str(KIC))
        plt.ylabel(r"$\nu$")
        plt.xlabel("Log Equivalent Duration")
        plt.yscale('log')

    errListUp = np.array([])
    errListDn = np.array([])
    totalEVFFitX = np.array([])
    totalEVFFitY = np.array([])
    quarterlyEVFX = []
    quarterlyEVFY = []
    time = np.array([])

    for x in range(fileCount):
        toteDuration = pd.read_table(files[x], skiprows=5, nrows=1, header=None, delim_whitespace=True, usecols=(7,)).iloc[0].values[0] #getting the total duration of each file
        df = pd.read_table(files[x], comment="#", delimiter=",", names=names)
        energy = np.array(df['Equiv_Dur']) #This is the energy column of the flare data
        positive = np.where(energy > 0)
        energy_p = energy[positive]
        sort = np.argsort(energy_p) #get indices that would sort the energy array
        ffdXEnergy = np.log10((energy_p + EPOINT)[sort][::-1])#log the reverse of sorted energy
        ffdYFrequency = (np.arange(1, len(ffdXEnergy)+1, 1))/toteDuration #get evenly spaced intervals, divide by totedur to get flares/day

        ok68 = (ffdXEnergy >= np.log10(np.median(df['ED68i'])) + EPOINT)

        if (any(ok68)):#taking care of the mean-fit data
            quarterlyEVFX.append(ffdXEnergy[ok68])
            quarterlyEVFY.append(ffdYFrequency[ok68])
            time = np.append(time, np.sum(df['t_start'])/len(df['t_start'])) #finding the mean time for a file

        if(kwargs['whole']==True): ok68 = np.isfinite(ffdXEnergy) #plotting all data

        totalEVFFitX = np.append(totalEVFFitX, ffdXEnergy[ok68])
        totalEVFFitY = np.append(totalEVFFitY, ffdYFrequency[ok68])
        errUp, errDn = calcError(ffdYFrequency[ok68], toteDuration)
        errListUp = np.append(errListUp, errUp)  #errup>>errDn

        if(PLOT):
            plt.plot(ffdXEnergy[ok68], ffdYFrequency[ok68], lw = 1, c = cmap(x/float(len(files))))
            if(kwargs['errore']==True): plt.errorbar(ffdXEnergy[ok68], ffdYFrequency[ok68], yerr = [errDn, errUp], c = 'black', elinewidth=.3, fmt='o', markersize = .55)





    sort = np.argsort(totalEVFFitX)
    if(len(totalEVFFitX) == 0):  return 0, 0, 0, 0, 0, 0, 0, False
    try:

        offset = min(totalEVFFitX) #starting the data at x=0 to reduce error in the the fit
        popt, pcov = curve_fit(power_law, totalEVFFitX[sort]-offset, totalEVFFitY[sort], p0=(.02, .4, .02),maxfev = 3000, sigma = errListUp)
        perr = np.sqrt(np.diag(pcov))


        if(PLOT):
            power_fit = power_law(totalEVFFitX[sort]-offset, *popt)
            positive = np.where(power_fit > min(totalEVFFitY))
            plt.plot(totalEVFFitX[sort][positive], power_fit[positive], c='black', lw=4, label="Best-Fit")
            plt.legend(loc='upper right')
            if(kwargs['save']==True): plt.savefig(bin_+'/'+evf_dir+'/'+ str(KIC) + '.png')
            if(kwargs['showe']==True):
                cbar = plt.colorbar(color, ticks=day_range)
                cbar.set_label('$BJD_{TDB}-2454832$', rotation=270)
                plt.show()
            plt.close()

        success = True

    except Exception as e:
        print("ERROR: Couldn't fit a power_law to " + KIC + ". Moving on to the next KIC")
        print("EXCEPTION: ",e)
        success = False
        popt = perr = offset = 0

    return quarterlyEVFX, quarterlyEVFY, popt, perr, toteDuration, time,offset, success





def plot_evf_sub(KIC, quarterlyEVFX, quarterlyEVFY, popt,perr,offset, toteDuration,bin_, **kwargs):
    if(PLOT):
        plt.figure()
        plt.title(str(KIC))
        plt.ylabel(r'$\nu$ - $\bar \nu$')
        plt.xlabel("Log Equivalent Duration")

    meanValues = np.array([])
    mean_errors = np.array([])
    for q in range(len(quarterlyEVFX)):
        fit = power_law(quarterlyEVFX[q]-offset, *popt)
        errUp, errDn = calcError(quarterlyEVFY[q], toteDuration)
        difference_err = calc_error_during_subtraction(quarterlyEVFX[q]-offset, errUp, popt, perr ,toteDuration)
        difference = quarterlyEVFY[q]-fit
        if(PLOT): plt.plot(quarterlyEVFX[q], difference, c = cmap(q/float(len(quarterlyEVFX))))
        #plt.errorbar(quarterlyEVFX[q], difference, yerr = [difference_err, difference_err], c = 'black', elinewidth=.6, fmt='o', markersize = 2, capsize=2)
        mean = np.mean(difference)
        mean_err = np.sqrt(np.sum(errUp**2))/np.size(errUp)
        meanValues = np.append(meanValues, mean)
        mean_errors = np.append(mean_errors, mean_err)

    if(PLOT):
        if(kwargs['save']==True): plt.savefig(bin_+'/'+evf_sub_dir+'/'+ str(KIC) + '.png')
        if(kwargs['showes']==True):
            cbar = plt.colorbar(color, ticks=day_range)
            cbar.set_label('$BJD_{TDB}-2454832$', rotation=270)
            plt.show()
        plt.close()

    return meanValues, mean_errors




def plot_evf_sub_mean(KIC, time, meanValues, mean_errors,  group_size, evf_sub_mean_data, targetIndex,bin_,**kwargs):
    if(group_size == 1):
        grouped_time = time
        grouped_mean_vals = meanValues
        grouped_mean_errs = mean_errors
    else:
        grouped_time = np.zeros([math.ceil(len(time)/group_size)])
        grouped_mean_vals = np.zeros([math.ceil(len(time)/group_size)])
        grouped_mean_errs = np.zeros([math.ceil(len(time)/group_size)])
        index = iterations = 0
        for x in range(len(time)):
            grouped_time[index] += time[x]
            grouped_mean_vals[index] += meanValues[x]
            grouped_mean_errs[index] += mean_errors[x]**2
            iterations += 1
            if((iterations == group_size) or (x == (len(time)-1))):
                grouped_time[index] = grouped_time[index]/iterations
                grouped_mean_vals[index] = grouped_mean_vals[index]/iterations
                grouped_mean_errs[index] = np.sqrt(grouped_mean_errs[index])
                iterations = 0
                index += 1

    fit_linspace = np.linspace(min(grouped_time), max(grouped_time), num=100)
    bestFit, bestParameters, bestCovariance, bestChiSquare, bestFitDegree, size, bics = compareFits(grouped_time, grouped_mean_vals, grouped_mean_errs, fit_linspace)
    evf_sub_mean_data = updateArray(evf_sub_mean_data, targetIndex, KIC, size, group_size, bestFitDegree, bestParameters, bestChiSquare, bestCovariance, bics)
    if(PLOT):
        plt.figure()
        plt.title(str(KIC))
        plt.ylabel(r'$\overline{\nu - \bar \nu}$')
        plt.xlabel("$BJD_{TDB}-2454832$")

        label = get_label(bestParameters, bestFitDegree, group_size)
        plt.scatter(grouped_time, grouped_mean_vals, c=grouped_time, cmap=cmap, lw=6)
        plt.plot(fit_linspace, bestFit, 'black', lw=4, label=label)
        if(kwargs['errorm']==True): plt.errorbar(grouped_time, grouped_mean_vals, yerr = grouped_mean_errs, c='black', fmt='o', markersize=0, elinewidth=.8, capsize=6)
        plt.legend(loc='upper right')


        if(kwargs['save']==True): plt.savefig(bin_+'/'+evf_sub_mean_dir+'/'+ str(KIC) + '_group_size_' + str(group_size) + '.png')
        if(kwargs['showm']==True): plt.show()
        plt.close()








def main():
    for bin_ in bin_list:
        file = bin_+'/'+'targets.txt'
        targetCount = getSize(file)
        print("Working on \'"+bin_+"\' which has a total of "+str(targetCount)+" targets.")
        ###################################################################################################################
        #THE EVF EVAL
        ###################################################################################################################
        targets = open(file, "r")
        grouping_nums = [1,2,3]
        evf_sub_mean_data, targetIndex = init_data_array(targetCount*len(grouping_nums) + 2)

        for line in targets:
            KIC = line.rstrip('\n')
            print("Working on the energy_vs_frequency analysis for KIC: "+str(KIC))
            files = sorted(glob('KICs/'+KIC+"/*.flare"))
            fileCount = len(files)

            quarterlyEVFX, quarterlyEVFY, popt, perr, toteDuration, time, offset, success  = plot_evf(KIC, files, fileCount,bin_, showe=SHOWE,errore=ERRORE,whole=WHOLE,save=SAVEPLOT)

            if(success):
                meanValues, mean_errors = plot_evf_sub(KIC, quarterlyEVFX, quarterlyEVFY, popt,perr, offset, toteDuration,bin_,showes=SHOWES, save=SAVEPLOT)
                for group_size in grouping_nums:
                    plot_evf_sub_mean(KIC, time, meanValues, mean_errors, group_size, evf_sub_mean_data, targetIndex,bin_,errorm=ERRORM,showm=SHOWM,save=SAVEPLOT)
                    targetIndex+= 1

        targets.close()
        if(SAVETXT==True): np.savetxt(bin_+'/'+fit_data_dir+'/evf_mean_sub.txt', evf_sub_mean_data, fmt = '% 20s', delimiter=' ', newline='\n', header='', footer='', comments='# ')

        ###################################################################################################################
        #THE TVF EVAL
        ###################################################################################################################
        energyConstantList = [1,2,3]
        for energyConstant in energyConstantList:

            targets = open(file, "r")
            fixedEnergy = energyConstant + EPOINT
            tvf_data, targetIndex = init_data_array(targetCount + 2)

            for line in targets:

                KIC = line.rstrip('\n')
                print("Working on the time_vs_frequency analysis for KIC: "+str(KIC)+ " at energy: "+str(fixedEnergy))
                files = glob('KICs/'+KIC+"/*.flare")
                fileCount = len(files)
                plot_tvf(KIC, files, fileCount, tvf_data, fixedEnergy, targetIndex, bin_, showt=SHOWT, errort = ERRORT,save=SAVEPLOT)
                targetIndex += 1

            targets.close()
            if(SAVETXT==True): np.savetxt(bin_+'/'+fit_data_dir+'/fixed_energy_equals_'+str(fixedEnergy)+'.txt', tvf_data, fmt = '% 20s', delimiter=' ', newline='\n', header='', footer='', comments='# ')


main()










'''

print("Input a .txt file containing the names of the KIC's you want to evaluate. After that, include:\n"+
"For the Energy_vs_frequency analysis:\n'-se': show plots\n'-w' : show the whole plot, including data below ok68 cutoff\n'-ee': include error bars in the plots\n\n"+
"For the Energy_vs_frequency mean-fit analysis:\n'-sm': show plots\n'-em': show error on the plots\n\n"+
"For the Time_vs_fruency analysis:\n'-st': show plots\n\n"+
"Use '-saveplot' to save all of the plots\n"+
"Use '-savefit' to save the fit data\n")


if (len(sys.argv)==1):
    print("ERROR: NO TARGET FILE INCLUDED")
    sys.exit()

try:
    file = sys.argv[1]
    targets = open(file, "r") # a file containing all the KICs we want to plot
except:
    print("\nERROR: Cannot open "+file+".")
    sys.exit()

SHOWE = check_args("-se")
ERRORE = check_args("-ee")
WHOLE = check_args("-w")
SHOWM = check_args("-sm")
ERRORM = check_args("-em")
SHOWT = check_args("-st")
SAVEPLOT = check_args("-saveplot")
SAVETXT = check_args("-savefit")

def check_args(key):
    for x in range(len(sys.argv)-2):
        if sys.argv[x+2]==key:
            return True
    return False
'''
