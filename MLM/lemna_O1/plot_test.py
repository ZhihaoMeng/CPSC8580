import random
import matplotlib  
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np

def draw_deduction(mat):
    fea_list=[5,15,25,35]
    method_name = ['lemna-no-GMM','lemna','lemna-customGmM','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Binary O1")
    plt.legend()
    plt.savefig('feature_deduction.png')
    plt.close()
def draw_synthetic(mat):
    fea_list=[5,15,25,35]
    method_name = ['lemna-no-GMM','lemna','lemna-customGMM','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Binary O1")
    plt.legend()
    plt.savefig('feature_synthetic.png')
    plt.close()
def draw_augmentation(mat):
    fea_list=[5,15,25,35]
    method_name = ['lemna-no-GMM','lemna','lemna-customGMM','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Binary O1")
    plt.legend()
    plt.savefig('feature_augmentation.png')
    plt.close()

if __name__ == '__main__':
#lemna-no-GMM pos 5,15,25,35
#lemna pos 5,15,25,35
#lemna-customGMM pos 5,15,25,35
#random pos 5,15,25,35
    mat1 = [[10.5035971223,7.62589928058,1.15107913669,0.647482014388],[4.74820143885,1.0071942446,1.15107913669,0.503597122302],[4.67625899281,0.863309352518,1.58273381295,0.575539568345],[99.2086330935,95.6834532374,93.7410071942,92.0863309353]]
    draw_deduction(mat1)

#lemna-no-GMM new 5,15,25,35
#lemna new 5,15,25,35
#lemna-customGMM new 5,15,25,35
#random new 5,15,25,35
    mat2 = [[96.8345323741,97.1223021583,99.6402877698,99.7122302158],[96.1151079137,99.4964028777,99.3525179856,99.7122302158],[96.4028776978,99.5683453237,99.2805755396,99.7122302158],[0.438848920863,11.0071942446,18.5611510791,22.4460431655]]
    draw_synthetic(mat2)

#lemna-no-GMM neg 5,15,25,35
#lemna neg 5,15,25,35
#lemna-customGMM neg 5,15,25,35
#random neg 5,15,25,35
    mat3 = [[70.3597122302,88.6330935252,97.5539568345,98.7050359712],[80.4316546763,95.7553956835,97.3381294964,98.7769784173],[82.4460431655,96.7625899281,96.9064748201,98.6330935252],[0.0719424460432,0.6474820143882,0.719424460432,2.01438848921]]
    draw_augmentation(mat3)