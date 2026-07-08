import yaml
import sys
import math

from ROOT import TGraphAsymmErrors, TFile

infile = sys.argv[1]

with open( infile,'r') as stream:
    try:
        ydict = yaml.safe_load(stream)
        xval  = [ v['value'] for v in ydict['independent_variables'][0]['values'] ]
        yval  = [ v['value'] for v in ydict['dependent_variables'][0]['values'] ]
        
        get_error = lambda v: math.sqrt( sum( w['symerror']**2 for w in v ) )
        
        yerr  = [ get_error( v['errors'] ) for v in ydict['dependent_variables'][0]['values'] ]
        
        result = TGraphAsymmErrors()
        
        for i in range(0,len(xval)):
            result.SetPoint(i,xval[i],yval[i])
            result.SetPointError(i,0,0,yerr[i],yerr[i])
        
        outfile  = infile.replace('.yaml','.root')
        rootfile = TFile( outfile, 'RECREATE' )
        result.Write('R')
        rootfile.Close()
        
    except yaml.YAMLError as exc:
        print(exc) 
