#
# circle detection for final project
#
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import Random
import random
import copy
from operator import attrgetter
import math
import sys
import optparse
import yaml
import time
    
def edge_detection(img,i):  
    
    im = cv2.imread(img)
    im = cv2.GaussianBlur(im, (3+2*i,3+2*i), 0)
    canny = cv2.Canny(im, 30, 200)
    cv2.imwrite('canny.jpg', canny)    
    return canny

def preprocess(mat):
    
    A=list(np.where(mat==255))
    x=A[1]
    y=A[0]
    white_points_cor=[]
    for i in range(len(A[0])):
        point=[x[i],y[i]]
        white_points_cor.append(point) 
    return white_points_cor


def circle_trans(A): 
    
    x1=A.circle_points_XY[0][0]
    y1=A.circle_points_XY[0][1]    
    x2=A.circle_points_XY[1][0]
    y2=A.circle_points_XY[1][1]   
    x3=A.circle_points_XY[2][0]
    y3=A.circle_points_XY[2][1]
    a=(y1-y2)*((x2**2)-(x3**2)+(y2**2)-(y3**2))
    b=(y2-y3)*((x1**2)-(x2**2)+(y1**2)-(y2**2))
    c=2*(x2-x3)*(y1-y2)-2*(x1-x2)*(y2-y3)
    if c==0: x0=0
    else: x0=(a-b)/c    
    a=(x1-x2)*((x2**2)-(x3**2)+(y2**2)-(y3**2))
    b=(x2-x3)*((x1**2)-(x2**2)+(y1**2)-(y2**2))
    c=2*(x2-x3)*(y2-y1)+2*(x1-x2)*(y2-y3)
    if c==0: y0=0
    else: y0=(a-b)/c
    r=(((x0-x1)**2)+((y0-y1)**2))**0.5
    return [x0,y0,r]


def points_in_cir(A):
    xi=[]
    yi=[]
    for i in range(A.Ns):
        xi.append(A.circle[0]+A.circle[2]*np.cos(2*np.pi*i/A.Ns))
        yi.append(A.circle[1]+A.circle[2]*np.sin(2*np.pi*i/A.Ns))
    points=list(zip(xi,yi)) 
    return points

def bin_to_dec(bin_str):
    bin = [int(n) for n in bin_str ]
    dec = [bin[-i - 1] * math.pow(2, i) for i in range(len(bin))]
    return int(sum(dec))

def dec_to_bin(dec,sup):
    bin = []
    while dec / 2 > 0:
        bin.append(str(dec % 2))
        dec = dec // 2    
    bin.reverse()    
    return ''.join(bin).zfill(sup)

def bin_to_declist(bin_str,sup):
    i1=bin_str[0:sup]
    i2=bin_str[sup:sup*2]
    i3=bin_str[sup*2:sup*3]
    return i1,i2,i3    
                
# population===================================================================    
class Population:
     crossoverfraction=None
     def __init__(self,populationsize,white_points_cor):
         self.population=[]
         for i in range(populationsize):
             self.population.append(individual(white_points_cor)) 
             
     def sum_fit(self):
         sum_f=0
         for ind in self.population:
             sum_f=sum_f+ind.fit
         return sum_f         
                
     def crossover(self):
         
        list1=list(range(len(self.population)))
        random.shuffle(list1)
        list2=list(range(len(self.population)))
        random.shuffle(list2)        
        for i in range(len(list1)):
            if list1[i] == list2[i]:
                temp=list2[i]
                if i == 0:
                    list2[i]=list2[-1]
                    list2[-1]=temp
                else:
                    list2[i]=list2[i-1]
                    list2[i-1]=temp

        for index1,index2 in zip(list1,list2):
            if random.random() < self.crossoverfraction:       
                self.population[index1].crossover(self.population[index2])


                           
     def mutate(self):     
         for individual in self.population:
             individual.mutate() 
             
     def combinePops(self,otherPop):
         self.population.extend(otherPop.population)
         
     def copy(self):
         return copy.deepcopy(self)  
     
     def truncateSelect(self,newPopSize):
         self.population.sort(key=attrgetter('fit'),reverse=True)
         self.population=self.population[:newPopSize]
           
     def evaluateFitness(self):
         for individual in self.population: individual.evaluateFitness()   

         
# individual===================================================================        
class individual:
    sup=None
    uniprng=None
    canny=None
    white_points_cor=None
    mutationrate=None
    def __init__(self,white_points_cor):
        self.mutRate=self.uniprng.uniform(0,1)
        self.gene=''
        for i in range(3):
            random_num=random.randint(0,len(white_points_cor)-1)
            self.gene=self.gene+dec_to_bin(random_num,self.sup)
       
        A=bin_to_declist(self.gene,self.sup)
        self.circle_points_XY=[]
        for i in range(3):
            self.circle_points_XY.append(white_points_cor[bin_to_dec(A[i])])
        self.circle=circle_trans(self) 
        self.Ns=int(np.round(0.1*self.circle[2]*2*np.pi))
        self.points_in_circle=points_in_cir(self)
        self.fit=fit.fitness(self,self.canny)
           
    def crossover(self,other):
        ran=random.randint(1,self.sup*3-1)
        temp1_self=self.gene[0:ran]
        temp2_self=self.gene[ran:self.sup*3]
        temp1_other=other.gene[0:ran]
        temp2_other=other.gene[ran:self.sup*3]
        
        self.gene=temp1_self+temp2_other
        other.gene=temp1_other+temp2_self       

            

        self.circle_points_XY=None
        other.circle_points_XY=None
        self.fit=None
        other.fit=None       
        self.circle=None
        other.circle=None     
        self.Ns=None
        other.Ns=None         
        self.points_in_circle=None
        other.points_in_circle=None

       
    def mutate(self):
        if self.mutRate < self.mutationrate:

            ran=random.randint(0,self.sup*3-1)
            A=list(self.gene)            
            if A[ran]=='0': A[ran]='1'
            else: A[ran]='0'            
            self.gene=''.join(A)
        
        self.mutRate = self.uniprng.uniform(0,1)
        self.circle_points_XY=None
        self.fit=None       
        self.circle=None 
        self.Ns=None        
        self.points_in_circle=None

       
    def evaluateFitness(self):
        if self.fit == None:
            A=bin_to_declist(self.gene,self.sup)
            self.circle_points_XY=[]
            for i in range(3):
                a=bin_to_dec(A[i]) 
                
    
                while a > len(self.white_points_cor)-1:
                    
                    a=a-len(self.white_points_cor) 
                
                self.circle_points_XY.append(self.white_points_cor[a])            
            self.circle=circle_trans(self)
            self.Ns=int(np.round(0.1*self.circle[2]*2*np.pi))
            self.points_in_circle=points_in_cir(self)
            self.fit=fit.fitness(self,self.canny)
        
    def __str__(self):
        return str(self.circle_points_XY)+'\t'+'%0.8e'%self.fit +'\t'+'%0.8e'%self.mutRate  

            
# ============================================================================= 
class fit:
    small_r=None
    
    @classmethod            
    def fitness(cls,A,canny):        
        y_max,x_max=np.shape(canny)
        i=0
        testx=0
        testy=0    
        for n in range(A.Ns):
            x_cor=int(np.round(A.points_in_circle[n][0]))
            y_cor=int(np.round(A.points_in_circle[n][1]))
            if testx==x_cor and testy==y_cor: i=i+0
            else:
                if x_cor >= x_max or y_cor >= y_max: i=i+0
                elif x_cor < 0 or y_cor < 0: i=i+0
                elif canny[y_cor,x_cor] == 255: i=i+1 
                else: i=i+0
            testx=x_cor
            testy=y_cor
        if A.circle[2]<cls.small_r: 
            if A.Ns==0: mod_fit=0
            else:   mod_fit=i/A.Ns*A.circle[2]/cls.small_r*0.7
        if A.circle[2]>=cls.small_r: mod_fit=i/A.Ns
        return mod_fit

class EV3_Config:
    # class variables
    sectionName='circle detection'
    options={'populationsize': (int,True),
             'generationlimit': (int,True),
             'crossoverfraction': (float,True),
             'mutationrate': (float,True),
             'testtime': (int,True),
             'img': (str,True),
             'binarylength': (int,True),
             'thresholdfitness': (float,True)} 
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV2 section
        infile=open(inFileName,'r')
        ymlcfg=yaml.load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))
         
        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]
 
                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                 
                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
     
    #string representation for class data    
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))
       
def printStats(pop,gen):
    print('Generation:',gen)
    avgval=0
    maxval=pop[0].fit 
    mutRate=pop[0].mutRate
    for ind in pop:
        avgval+=ind.fit
        if ind.fit > maxval:
            maxval=ind.fit
            mutRate=ind.mutRate
        print(ind)

    print('Max fitness',maxval)
    print('MutRate',mutRate)
    print('Avg fitness',avgval/len(pop))
    print('')
    return maxval,avgval/len(pop)

    
def circledetection(cfg):

    pixel_num=300
    individual.sup=cfg.binarylength
    fit.small_r=pixel_num/10
    individual.mutationrate=cfg.mutationrate
    Population.crossoverfraction=cfg.crossoverfraction
    individual.uniprng=Random()
    times=[]
    each_ans=[]
    for number_success in range(cfg.testtime): 
        start=time.time()        
        max_val=[]
        avg_val=[]
        A='1'
        for i in range(individual.sup-1):
            A=A+'1'
        limit_pix=bin_to_dec(A)          
        canny=edge_detection(cfg.img,0)
        white_points_cor=preprocess(canny)
        i=1   

        while len(white_points_cor) > limit_pix:
            canny=edge_detection(cfg.img,i) 
            white_points_cor=preprocess(canny)
            i=i+1
        
        individual.canny=canny
        individual.white_points_cor = white_points_cor       
        pop=Population(cfg.populationsize,white_points_cor)        
        printStats(pop.population,0)        
        for i in range(cfg.generationlimit):    
        
            offspring=pop.copy()

            offspring.crossover()
        
            offspring.mutate()
            
            offspring.evaluateFitness()
                        
            pop.combinePops(offspring)
            
            pop.truncateSelect(cfg.populationsize)
             
            max_avg=printStats(pop.population,i+1)
            
            max_val.append(max_avg[0])
            avg_val.append(max_avg[1])
            
            if pop.population[0].fit >= cfg.thresholdfitness: break
        
            if pop.population[0].fit==pop.population[cfg.populationsize-1].fit:
                pop.truncateSelect(2)
                newpops=Population(cfg.populationsize-2,white_points_cor)
                pop.combinePops(newpops) 
        times.append(time.time()-start) 
        
        A=pop.population[0].circle
        x0=int(np.round(A[0]))
        y0=int(np.round(A[1]))
        r=int(np.round(A[2])) 
        each_ans.append([x0,y0,r])
        img_ori=cv2.imread(cfg.img)
        cv2.circle(img_ori,(x0,y0),r, (11,255,255),2)            
        cv2.imwrite('canny_final.jpg', img_ori)
        
        plt.figure(1)
        plt.title('max fitness - generation')
        plt.xlabel('generation')
        plt.ylabel('maximum fitness')
        plt.plot(max_val)

        plt.show()
    print('total time',sum(times))
    print('average time: ',sum(times)/cfg.testtime)
 
    
def main(argv=None):
    # if argv is None:
    #     argv = sys.argv        
    # try:
    #     parser = optparse.OptionParser()
    #     parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
    #     parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
    #     parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
    #     (options, args) = parser.parse_args(argv)

    #     if options.inputFileName is None:
    #         raise Exception("Must specify input file name using -i or --input option.")

    # cfg=EV3_Config(options.inputFileName)
    cfg=EV3_Config("config_2.cfg")

    print(cfg)

    circledetection(cfg)

    #     if not options.quietMode:                    
    #         print('EV3 Completed!')    
    
    # except Exception as info:
    #     if 'options' in vars() and options.debugMode:
    #         from traceback import print_exc
    #         print_exc()
    #     else:
    #         print(info) 
            
if __name__ == '__main__':
    main()
