#RouteNo=5
#distance=[16,20,16,24,20]
#peak hour:4
#off-peak hour: 14
#speed=[(40,48)] #peak hour vs. off-peak hour
#demand=[(200,100),(300,150),(150,60).(400,200),(250,125)] # peak hour vs. off-peak hour

import OneSizeSystemwide as oss
import TwosizeSystemwide as tss
import FOT as fot
import pickle
import os

import logging
import logging.config
import yaml

with open('logging.yaml','r',encoding='utf-8') as f:
    #config=yaml.load(f,Loader=yaml.FullLoader)
    config = yaml.safe_load(f)
    #print(config)
    logging.config.dictConfig(config)
logger=logging.getLogger('mylogger')

def load_pickle(path):
    with open(path,'rb') as f:
        while 1:
            try:
                y=pickle.load(f)
                yield y
            except EOFError:
                logger.error('Error', exc_info=True)
                break

def Freight_Systemwide(routeNo,distance,average_distance,speed,demand,peak_point_demand,type):
    typeList=['OneS','TwoS','TwoSF']
    result=None
    if type not in typeList:
        logger.debug('The type is wrong, please reset it.')
    elif type==typeList[0]:
        case1 = oss.OneSize(routeNo, distance, average_distance, speed,demand, peak_point_demand)
        result1 = case1.Optimal()
        result={}
        result['objval']=result1[0]
        result['result']=result1[1]#[c_o, c_uw, c_uv, c_u, c_total, c_p]
        result['S']=result1[2]
        result['fleet_size']=result1[3]
        result['bus_operating_cost']=result1[4]
        result['h_jt']=result1[5]
        result['h_jt_1']=result1[6]
        result['h_jt_2']=result1[7]
        result['c_o_jt']=result1[8]
        result['c_uw_jt']=result1[9]
        result['c_uv_jt']=result1[10]
        result['c_u_jt']=result1[11]
        result['c_jt']=result1[12]
        result['n_jt']=result1[13]
        result['n_t']=result1[14]
        with open('data_one.pickle','wb') as f:
            pickle.dump(result,f)
        logger.info('One size systemwide finished!')
    elif type==typeList[1]:
        case2 = tss.TwoSize(routeNo, distance, average_distance, speed, demand,peak_point_demand, size_type=2)
        result = case2.BFGS()
        #one item in the result
        #iter_no, {"solution": x, "delta": delta_x, "headway": headway_x,"fleet": fleet_size_x,"objValue": ctp_x}
        with open('data_two.pickle','wb') as f:
            pickle.dump(result,f)
        logger.info('Two size systemwide finished! ')
    elif type==typeList[2]:
        case3=fot.FOT(routeNo, distance, average_distance, speed, demand, peak_point_demand)
        #generate y_initial
        if os.path.exists('data_one.pickle'):
            result_init=Freight_Systemwide(routeNo, distance, average_distance, speed, demand, peak_point_demand, 'OneS')
        else:
            result_init=load_pickle('data_one.pickle')
        period=len(demand[0])
        y_initial={}
        y_initial['N_hat']=result_init['n_jt']
        y_initial['N_s']={1:result_init['fleet_size'],2:0}
        y_initial['q']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['X']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['delta']={(j,t):1 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['xi']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['zeta']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}

        #result_s
        #y:
        # N_hat: N_j_t
        # N_bar: N_s
        # q: q_j_t
        # X: X_j_t
        # delta: delta_j_t
        # xi: xi_j_t
        # zeta: zeta_j_t
        result=case3.solveBenders(100,y_initial,1000)#result_s,y,UB_LB_tol_dict
                                                                #UB_LB_tol_dict:{iter:(UB,LB,UB-LB)}
    return result

if __name__=="__main__":
    distance = [16, 20, 16, 24, 20]
    average_distance = [5, 7, 5, 8, 6]
    speed=[
        [48,40,40,48,48,48,48,48,48,48,48,40,40,48,48,48,48,48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48]
    ]
    demand=[
        [100,200,200,100,100,100,100,100,100,100,100,200,200,100,100,100,100,100],
        [150, 300, 300, 150, 150, 150, 150, 150, 150, 150, 150, 300, 300, 150, 150, 150, 150, 150],
        [60, 150, 150, 60, 60, 60, 60, 60, 60, 60, 60, 150, 150, 60, 60, 60, 60,60],
        [200, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200, 400, 400, 200, 200, 200, 200, 200],
        [125, 250, 250, 125, 125, 125, 125, 125, 125, 125, 125, 250, 250, 125, 125, 125, 125, 125]
    ]
    peak_point_demand=[
        [100,200,200,100,100,100,100,100,100,100,100,200,200,100,100,100,100,100],
        [150, 300, 300, 150, 150, 150, 150, 150, 150, 150, 150, 300, 300, 150, 150, 150, 150, 150],
        [60, 150, 150, 60, 60, 60, 60, 60, 60, 60, 60, 150, 150, 60, 60, 60, 60,60],
        [200, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200, 400, 400, 200, 200, 200, 200, 200],
        [125, 250, 250, 125, 125, 125, 125, 125, 125, 125, 125, 250, 250, 125, 125, 125, 125, 125]
    ]

    '''
    #OneSizeSystemwide
    #def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand):
    case1=oss.OneSize(routeNo=5,distance=[16,20,16,24,20],average_distance=[5,7,5,8,6],speed=speed,demand=demand,peak_point_demand=peak_point_demand)
    result1=case1.Optimal()
    print(result1)
    '''

    '''
    #TwoSizeSystemwide
    #def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand,size_type):
    case2=tss.TwoSize(routeNo=5,distance=distance,average_distance=average_distance,speed=speed,demand=demand,peak_point_demand=peak_point_demand,size_type=2)
    # can not run #
    #result2=case2.Optimal()
    result2=case2.BFGS()
    print(result2[len(result2)-1]['solution'])
    print(result2[len(result2) - 1]['objValue'])
    print(result2[len(result2) - 1]['headway'])
    print(result2[len(result2) - 1]['delta'])
    print(result2[len(result2) - 1]['fleet'])
    print(result2[0]['solution'])
    print(result2[0]['objValue'])

    pass
    '''

    #typeList=['OneS','TwoS','TwoSF']
    result_oneSize=Freight_Systemwide(5, distance, average_distance, speed, demand, peak_point_demand, 'OneS')
    print(result_oneSize)
    logger.info('The result of one size is\n {}'.format(result_oneSize))
    result_twoSize = Freight_Systemwide(5, distance, average_distance, speed, demand, peak_point_demand, 'TwoS')
    print(result_twoSize[len(result_twoSize)-1])
    logger.info('The result of two-size systemwide is \n {}'.format(result_twoSize[len(result_twoSize)-1]))
    result_twoSizeF=Freight_Systemwide(5, distance, average_distance, speed, demand, peak_point_demand, 'TwoSF')
    print(result_twoSizeF)
    logger.info('The result of two-size systemwide with freight on transit is \n {}'.format(result_twoSizeF))
