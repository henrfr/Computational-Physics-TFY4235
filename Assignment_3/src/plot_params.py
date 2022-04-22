import matplotlib.pyplot as plt

def set_plot_parameters(size: int=16) -> None:
    """
    A helper function for setting plot parameters.
    """
    plt.style.use('seaborn-bright')
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rc('legend',fontsize=size-4)
    font = {'family' : 'serif', 
            'size': size}
    plt.rc('font', **font)
    plt.rc('lines', lw=2)