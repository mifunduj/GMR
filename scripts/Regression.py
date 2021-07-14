import csv
import argparse
import numpy as np
import GaussianMixtureRegression as gmr
from sklearn.mixture import GaussianMixture as GM

def main():
    epilog=""""

    """""
    parser=argparse.ArgumentParser(description='',epilog=epilog)
    parser.add_argument('-f','--file',
                required=True, help='file containing recorded data')
    parser.add_argument('-n','--number',type=int,
                required=True, help=' number of files containing recorded data')
    parser.add_argument('-np','--n_parameters',default=17,
                type=int, help='number of parameters: 1 + n_input + n_output')
    parser.add_argument('-i','--n_input',default=1,
                    type=int, help='number of parameters in input')
    parser.add_argument('-nb','--n_begin',default=10,type=int,
                help='first number of components during reshearch of best number of components')
    parser.add_argument('-ne','--n_end',default=30,type=int,
                help='last number of components during reshearch of best number of components')
    parser.add_argument('-s','--sigma_filtre',default=.4,type=float,
                help='variance for the gaussian filter')
    args=parser.parse_args()
#------------------------------------------------------------------------------------------------
    n_out=args.n_parameters-args.n_input
    data=gmr.fuse_data(args.file,args.number,args.n_parameters)
    data_f=gmr.gaussian_filter(data,args.sigma_filtre)
    best_value=gmr.best_n_components(data_f,n_out,args.n_begin,args.n_end)
    gmm=GM(n_components=best_value)
    gmm.fit(data_f,data_f[:,args.n_input::])

    if args.n_input==1:
        r=gmr.regression(gmm,data_f[:,0])
    else:
        end_=args.n_input-1
        r=gmr.regression(gmm,data_f[:,0:end_])
    file_name=args.file + '_regression'
#    _header=[]
    gmr.write_file(file_name,r)
#    gmr.write_file(file_name,r,_header_=_header)

if __name__=='__main__':
    main()
