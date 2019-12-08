import random
import matplotlib  
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np

def draw_deduction(mat):
    fea_list=[5,15,25,35]
    method_name = ['lemna-no-GMM','lemna','lemna-customGMM','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Binary O0")
    plt.legend()
    plt.savefig('feature_deduction_O0.png')
    plt.close()
def draw_synthetic(mat):
    fea_list=[5,15,25,35]
    method_name = ['lemna-no-GMM','lemna','lemna-customGMM','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Binary O0")
    plt.legend()
    plt.savefig('feature_synthetic_O0.png')
    plt.close()
def draw_augmentation(mat):
    fea_list=[5,15,25,35]
    method_name = ['lemna-no-GMM','lemna','lemna-customGMM','random']
    for i,method in enumerate(mat):
        plt.plot(fea_list,method,"x-",label=method_name[i])  
        plt.scatter(fea_list,method,marker='x',s=30) 
    plt.xlabel('Nfeature')
    plt.ylabel('PCR(%)')
    plt.title("Binary O0")
    plt.legend()
    plt.savefig('feature_augmentation_O0.png')
    plt.close()

if __name__ == '__main__':
#lemna-no-GMM pos 5,15,25,35
#lemna pos 5,15,25,35
#lemna-customGMM pos 5,15,25,35
#random pos 5,15,25,35
    mat1 = [[24.1903827282,15.7507360157,6.67320902846,3.87634936212],[9.4700686948,13.1010794897,5.8390578999,4.5142296369],[7.99803729146,13.8861629048,7.01668302257,3.72914622179],
[99.9018645731,99.4111874387,98.5279685967,97.6447497547]]
    draw_deduction(mat1)

#lemna-no-GMM new 5,15,25,35
#lemna new 5,15,25,35
#lemna-customGMM new 5,15,25,35
#random new 5,15,25,35
    mat2 = [[87.2914622179,91.0696761531,97.350343474,99.8037291462],[95.2404317959,96.3199214917,98.4789008832,99.9018645731],[95.3876349362,95.1913640824,97.4975466143,99.9509322866],[9.12659470069,15.6526005888,23.7978410206,27.6251226693]]
    draw_synthetic(mat2)

#lemna-no-GMM neg 5,15,25,35
#lemna neg 5,15,25,35
#lemna-customGMM neg 5,15,25,35
#random neg 5,15,25,35
    mat3 = [[67.8606476938,83.6113837095,93.375858685,97.791952895],[83.3660451423,87.8312070658,95.4367026497,98.233562316],[85.9666339549,86.8007850834,93.5230618253,97.8900883219],[0,0.0490677134446,0.686947988224,1.47203140334]]
    draw_augmentation(mat3)