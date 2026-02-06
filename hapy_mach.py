import os, subprocess as sp
#---------------------------------------------------------------------------------------------------
def get_host(verbose=False):
    """
    Get name of current machine
    """
    # First get some info
    try: 
        host = sp.check_output(["dnsdomainname"],universal_newlines=True).strip()
    except:
     host = None
    if host=='chn': host = 'nersc' # reset for perlmutter
    if host is not None:
        if 'nersc' in host : host = None
        if host is None or host=='' : host = os.getenv('NERSC_HOST')
    if host is None or host=='' : host = os.getenv('host')
    if host is None or host=='' : host = os.getenv('HOST')
    opsys = os.getenv('os')
    #-----------------------------------------------------------------------------
    if verbose:
        print()
        print('\n'+f'  host : {host}')
        print(     f'  opsys: {opsys}\n')
    #-----------------------------------------------------------------------------
    # Make the final setting of host name
    if opsys=='Darwin'      : host = 'mac'
    if 'perlmutter' in host : host = 'nersc'
    if 'summit'     in host : host = 'olcf'
    if host=='ccs.ornl.gov' : host = 'olcf'   # rhea
    if host=='olcf.ornl.gov': host = 'olcf'   # andes
    if host=='lcrc.anl.gov' : host = 'lcrc'    # chrysalis
    if 'aurora.alcf' in host: host = 'alcf'
    return host
#---------------------------------------------------------------------------------------------------