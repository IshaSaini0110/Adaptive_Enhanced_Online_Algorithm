from pylab import *
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import csv
import random
import numpy as np
import pandas as pd
import seaborn as sns
mpl.rcParams['font.family'] = ['Times New Roman']

#Big data analysis job generator, generated based on real data sets
def TaskGenerator(n):
    TaskList = []
    AcutualList = []
    #with open('C:\\Users\\KZ\\Desktop\\google cluster\\data\\output0.csv', 'r') as csvfile:
    with open('D:\Journal Online algo\output.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        time=[]
        for row in reader:
            # if(int(row[2])>=10 and int(row[2])<=195):
            #if(int(row[2])>=195 and int(row[2])<=545):
            # if(int(row[2])>=300):
            # if(int(row[2])>=545):
            if(int(row[2]) >=100):
                time.append(int(row[2]))
        csvfile.close()
        for i in range(n):
            #Randomly take a number from time and delete it with pop
            #actual = time.pop(random.randint(0, len(time)-1))
            actual = time[random.randint(0, len(time) - 1)]
            down = actual - random.randint(0, int(0.5*actual))
            if down<0:
                down=1
            up = actual + random.randint(0, int(0.5*actual))
            TaskList.append([down, up, actual])
            AcutualList.append(actual)
    return TaskList,AcutualList

# #Task list and actual running time list
# TL,AL=TaskGenerator(100)
# print(TL)
# print(AL)

#Intermediate algorithm, all jobs run serially first, and when the number of idle machines reaches a, all tasks are completed in parallel.
def intermediate(AL,m,a,PF,addPF,SCPF):
    AL.sort()
    RunningTime = 0
    tempAL = []
    flag=AL[a-1]
    RunningTime+=flag
    for i in range(len(AL)):
        if i<a:
            tempAL.append(AL[i])
        else:
            if (AL[i]-flag) == 0:
                tempAL.append(AL[i])
            if (AL[i]-flag) != 0:
                #Parralel penalty for calculation of part exceeding flag
                tempAL.append(flag+(AL[i]-flag)*(PF+addPF*(m-1))/m)
                RunningTime+=((tempAL[i]-flag)+AL[i]*SCPF)
    # print(tempAL)
    print('Intermediate Algorithm Final Runtime:'+str(RunningTime))
    return RunningTime
#intermediate(AL,100,10,1,0.05)

#Approximately optimal algorithm, knowing the running time and arrival time of all jobs, calculate the minimum time required to complete this batch of jobs 
def optimal(AL,PF,addPF,SCPF):
    AL.sort()
    RunningTime = 0
    Task=[]
    tempAL=[]
    index=0
    #Stop allocating empty machines
    STOP = False
    #Initialize the job list. Task[0]：the actual running time of the job；Task[1]：which machines the job currently occupies；Task[2]：the current penalty factor of the job
    for i in AL:
        Task.append([i,[index],PF])
        tempAL.append(i)
        index+=1
    tempMin=min(tempAL)
    #One traversal means that the machine is idle once
    while sum(tempAL)!=0:
        #When the job with the shortest runtime finishes, a machine becomes idle. At this point, the scheduling strategy should be evaluated: if there are idle machines, it indicates that the overhead of parallelization outweighs the time saved by parallel execution.
        for i in range(len(tempAL)):
            tempAL[i] -= tempMin
            if tempAL[i]<0:
                tempAL[i]=0
        RunningTime += tempMin
        # print(Task)
        # print(tempAL)
        #Each traversal allocates an empty machine
        while min(tempAL)==0:
            tempMAX = max(tempAL)
            MaxIndex = tempAL.index(tempMAX)
            #Clearly know how many idle machines are there
            emptymachine=[]
            for i in range(len(tempAL)):
                if tempAL[i]==0:
                    emptymachine.append(i)
            #If all are idle, it means that all jobs have finished running and the loop is exited
            if len(emptymachine) == len(tempAL):
                break
            #Judge the longest running job: Can allocating more machines to it shorten the total running time ?
            if tempMAX*len(Task[MaxIndex][1])/Task[MaxIndex][2]*(Task[MaxIndex][2]+addPF*tempAL.count(0))/(tempAL.count(0)+len(Task[MaxIndex][1]))+SCPF*Task[MaxIndex][0] >=tempMAX:
                tempMin = min(filter(lambda x: x > 0, tempAL))
                STOP=True
                break
            mac=[]
            for machine in range(tempAL.count(0)):
                if tempMAX*len(Task[MaxIndex][1])/Task[MaxIndex][2]*(Task[MaxIndex][2]+addPF*(machine+1))/((machine+1)+len(Task[MaxIndex][1]))+SCPF*Task[MaxIndex][0] <tempMAX:
                    mac=machine+1
                    STOP = False
                    break
            #Assign machines: Penalty factor increases,machine running jobs updated
            oldtime=tempMAX*len(Task[MaxIndex][1])/Task[MaxIndex][2]
            Task[MaxIndex][1]+=emptymachine[0:mac]
            Task[MaxIndex][2] = PF+addPF*(len(Task[MaxIndex][1])-1)
            for i in Task[MaxIndex][1]:
                Task[i]=Task[MaxIndex]
                tempAL[i]=oldtime*Task[MaxIndex][2]/len(Task[MaxIndex][1])+SCPF*Task[MaxIndex][0]
            # print(Task)
            # print(tempAL)
        if not STOP:
            tempMin = min(tempAL)
    print('Optimal Algorithm Final Runtime:'+str(RunningTime))
    return RunningTime
#optimal(AL,1,0.05)

#Get the 4th element of the collection
def fourth(elem):
    return elem[3]
#Online Algorithm, cyclic allocation of idle resources
def online(TL,PF,addPF,SCPF,line):
    RunningTime = 0
    Task = []
    tempAL = []
    #loop list 
    L=[]
    point=-1
    index = 0
    for i in TL:
        if i[0]==i[1]:
            value=(i[1]-line)*(i[0]+i[1])/1
        else:
            value = (i[1]-line)*(i[0]+i[1])/(i[1]-i[0])
        #Each task is a five-tuple: actual stand-alone running time, deployment machine encoding set, penalty factor, circular queue arrangement basis, and stand-alone running time upper limit
        Task.append([i[2],[index],PF,value,i[1]])
        index+=1
    #The circular queue arrangement here is based on: (upper limit - threshold) * (upper limit + lower limit) / (upper limit - lower limit), rearrange according to the circular queue sorting  basis
    Task.sort(key=fourth,reverse=True)
    for i in range(len(Task)):
        tempAL.append(Task[i][0])
        L.append(Task[i])
        Task[i][1]=[i]
    tempMin=min(tempAL)
    #Open traversal means that the machine is idle once
    while sum(tempAL) != 0:
        #The job with the shortest running time has finished and there are idle machines
        #As jobs complete, the actual runtime and the maximum runtime of other jobs decrease. This is because the elements in Task and L refer to the same objects, so when Task[i][4] decreases, the corresponding value in L also decreases
        for i in range(len(tempAL)):
            tempAL[i] -= tempMin
            Task[i][4] -= tempMin
            if tempAL[i] < 0:
                tempAL[i] = 0
            if Task[i][4] < 0:
                Task[i][4] = 0
        #If the maximum running time of a job is less than the line threshold, then delete it from L.If the job before the pointer is deleted, the pointer moves forward accordingly
        for i in L:
            if i[4]<=line:
                if L.index(i)<point:
                    point-=1
                L.remove(i)
        RunningTime += tempMin
        if sum(tempAL) == 0:
            print('Online Algorithm Final Runtime:' + str(RunningTime))
            return RunningTime
        #If L is empty, all remaining jobs do not need to be run in parallel, and can be run serially until completion.
        if not L:
            RunningTime += max(tempAL)
            print('Online ALgorithm Final Runtime:' + str(RunningTime))
            return RunningTime
        # print(Task)
        # print(tempAL)
        # print(L)
        #Determine whether the penalty time of the last operation needs to be subtracted
        PunishorNot=[]
        for i in range(len(TL)):
            PunishorNot.append(0)
        #Each traversal allocates an empty machine
        while min(tempAL) == 0:
            tempMin = min(tempAL)
            MinIndex = tempAL.index(tempMin)
            #Delete the completed job from L. If the job before the pointer is deleted, the pointer will move forward accordingly
            if Task[MinIndex] in L:
                if L.index(Task[MinIndex]) < point:
                    point -= 1
                L.remove(Task[MinIndex])
                # If L is empty, all remaining jobs do  not need to be run in parallel and can be run serially until completion.
                if not L:
                    RunningTime += max(tempAL)
                    print('Online Algorithm Final Runtime:' + str(RunningTime))
                    return RunningTime
            #The pointer moves back and selects the next job in the circular queue
            point+=1
            #The pointer returns to its original position after one circle
            if point>=len(L)-1:
                point=0
            tempL=L[point]
            MaxIndex=Task.index(tempL)
            tempMAX=tempAL[Task[MaxIndex][1][0]]
            #Judgement: if the maximum running time of the selected job is greater than the threshold line, allocate one more machine to it
            if Task[MaxIndex][4]>line:
                # Assign machines, increase penalty factor, and update jobs running on each machine
                Task[MaxIndex][1].append(MinIndex)
                Task[MaxIndex][2] += addPF
                if PunishorNot[MaxIndex]==0:
                    Task[MaxIndex][4] = Task[MaxIndex][4] * Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                    PunishorNot[MaxIndex] = 1
                else:
                    Task[MaxIndex][4] = (Task[MaxIndex][4] - SCPF * Task[MaxIndex][0])*(len(Task[MaxIndex][1])-1)/(Task[MaxIndex][2]-addPF)* Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                for i in Task[MaxIndex][1]:
                    Task[i] = Task[MaxIndex]
                # Job running time update
                tempAL[MinIndex] = tempMAX * Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                tempAL[MaxIndex] = tempMAX * Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                # print(Task)
                # print(tempAL)
                # print(L)
            #If the selected job's maximum runtime is not greater than the threshold `line`, it is removed from `L`. If a job before the current pointer is deleted, the pointer is moved accordingly.
            else:
                if L.index(Task[MaxIndex])<point:
                    point-=1
                L.remove(Task[MaxIndex])
                #If L is empty, all remaining jobs do not need to be run in parallel and can be run serailly until completion.
                if not L:
                    RunningTime += max(tempAL)
                    print('Online Algorithm Final Runtime:' + str(RunningTime))
                    return RunningTime
            tempMin = min(tempAL)
    return RunningTime
#online(TL,1,0.05,30)

def adaptive_enhanced_online(TL, PF, addPF, SCPF, line):
    # Represent each task with original time, remaining time, allocated cores, and overhead count.
    jobs = []
    for lb, ub, actual in TL:
        jobs.append({'original': actual, 'remaining': actual, 'allocated': 1, 'overhead': 0})
    # Repeatedly assign extra cores to the job with highest positive benefit.
    while True:
        best_job = None
        best_benefit = 0.0
        best_predicted = 0.0
        for job in jobs:
            # Skip finished jobs or those not eligible (below threshold).
            if job['remaining'] <= 0 or job['original'] <= line:
                continue
            k = job['allocated']
            predicted = job['remaining'] * (k/(k+1)) * (1 + addPF)
            benefit = (job['remaining'] - predicted) - SCPF * job['original']
            if benefit > best_benefit:
                best_benefit = benefit
                best_job = job
                best_predicted = predicted
        if best_job is None or best_benefit <= 0:
            break
        # Allocate one more core to the best job.
        best_job['allocated'] += 1
        best_job['remaining'] = best_predicted
        best_job['overhead'] += 1
    # After allocation, compute each job's final completion time including overhead.
    final_times = []
    for job in jobs:
        final_times.append(job['remaining'] + job['overhead'] * SCPF * job['original'])
    total_runtime = max(final_times) if final_times else 0.0
    print(f"Adaptive Enhanced Online Algorithm Final Runtime: {total_runtime}")
    return total_runtime

def Line_picmaker_enhanced(line, length):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # New list for enhanced algorithm
    opt = []

    for i in range(length):
        # m = 57
        # m = random.randint(10, 20)
        m=random.randint(10, 1000)
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000

        TL, AL = TaskGenerator(m)
        
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
        IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
        IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
        
        opt.append(OPT)
        
        # Calculate ratios for comparison
        # im.append(1 if OPT/IM > 1 else OPT/IM)
        # im1.append(1 if OPT/IM1 > 1 else OPT/IM1)
        # im2.append(1 if OPT/IM2 > 1 else OPT/IM2)
        # ol.append(1 if OPT/OL > 1 else OPT/OL)
        # enhanced.append(1 if OPT/ENHANCED > 1 else OPT/ENHANCED)

        if (OPT / IM > 1):
            im.append(1)
        else:
            im.append(OPT / IM)

        if (OPT / IM1 > 1):
            im1.append(1)
        else:
            im1.append(OPT / IM1)

        if (OPT / IM2 > 1):
            im2.append(1)
        else:
            im2.append(OPT / IM2)

        if (OPT / OL > 1):
            ol.append(1)
        else:
            ol.append(OPT / OL)

        if (OPT / ENH > 1):
            enh.append(1)
        else:
            enh.append(OPT / ENH)

    
    x = range(length)
    plt.figure(figsize=(10, 3))
    plt.plot(x, im, color='#FF8C00', label='Two-phase algorithm (a=0.2m)', marker='^', markersize=3)
    plt.plot(x, im1, color='#00BFFF', label='Two-phase algorithm (a=0.5m)', marker='v', markersize=3)
    plt.plot(x, im2, color='#808080', label='Two-phase algorithm (a=0.8m)', marker='s', markersize=3)
    plt.plot(x, ol, color='r', label='Online algorithm', marker='.', markersize=3)
    plt.plot(x, enh, color='#32CD32', label='Adaptive Enhanced Online', marker='o', markersize=3)
    
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.25),
    ncol=3,                # 3 cols → first row 3 entries, second row 2 entries
    columnspacing=1.2,     # horiz space between columns
    handletextpad=0.5,     # space between marker and label text
    borderaxespad=0,       # pad between axes and legend
    fontsize=12,
    frameon=False)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    plt.savefig('small.pdf',dpi=600,format='pdf',bbox_inches = 'tight')
    plt.savefig('big.pdf',dpi=600,format='pdf',bbox_inches = 'tight')
    plt.savefig('bad1.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig('bad2.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
Line_picmaker_enhanced(100, 100)    

# (Bar Chart) - Enhanced
def Bar_picmaker_enhanced(line, time):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # New list for enhanced algorithm
    opt = []

    # m = 57
    # m = random.randint(10, 20)
    m=random.randint(10, 1000)
    TL, AL = TaskGenerator(m)
    bar = []
    
    for i in range(time):
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000
        
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
        IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
        IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
        
        opt.append(OPT)
        ol.append(OL)
        enh.append(ENH)
        im.append(IM)
        im1.append(IM1)
        im2.append(IM2)
        bar.append(i)
    
    bar_width = 0.12
    index_opt = np.arange(len(bar))
    index_OL = index_opt + bar_width
    index_enh = index_opt + bar_width*2
    index_IM = index_opt + bar_width*3
    index_IM1 = index_opt + bar_width*4
    index_IM2 = index_opt + bar_width*5
    
    
    plt.figure(figsize=(12, 5))
    plt.bar(index_opt, height=opt, width=bar_width, color='k', label='Offline optimal algorithm')
    plt.bar(index_OL, height=ol, width=bar_width, color='r', label='Online algorithm')
    plt.bar(index_enh, height=enh, width=bar_width, color='#32CD32', label='Adaptive enhanced online')
    plt.bar(index_IM, height=im, width=bar_width, color='#FF8C00', label='Two-phase algorithm (a=0.2m)')
    plt.bar(index_IM1, height=im1, width=bar_width, color='#00BFFF', label='Two-phase algorithm (a=0.5m)')
    plt.bar(index_IM2, height=im2, width=bar_width, color='#808080', label='Two-phase algorithm (a=0.8m)')
    
    
    plt.xticks(index_opt + bar_width*5 / 2, bar)
    plt.yticks(fontsize=14)
    plt.ylabel('Time (second)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Groups', fontsize=14)
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.20),
    ncol=3,
    borderaxespad=0,
    columnspacing=1.2,    # ↑ more space between columns
    # handlelength=2.5,     # ↑ longer color‐box (makes it less cramped)
    handletextpad=0.5,    # ↑ more gap between swatch and label     # controls vertical space between rows
    fontsize=12,
    frameon=False)
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.savefig(f'm={m}_enhanced.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
for i in range(3):
    Bar_picmaker_enhanced(100, 15)    

#(Box Plot) - Enhanced
def Box_picmaker_enhanced(line, time):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # New list for enhanced algorithm
    opt = []

    # m = 57
    # m = random.randint(10, 20)
    m=random.randint(10, 1000)
    PF = 1
    addPF = (random.randint(1, 500)) / 1000
    SCPF = (random.randint(1, 500)) / 1000

    for i in range(time):
        TL, AL = TaskGenerator(m)
        
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
        IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
        IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
        
        
        opt.append(OPT)
        
        # Calculate ratios for comparison
        im.append(1 if OPT/IM > 1 else OPT/IM)
        im1.append(1 if OPT/IM1 > 1 else OPT/IM1)
        im2.append(1 if OPT/IM2 > 1 else OPT/IM2)
        ol.append(1 if OPT/OL > 1 else OPT/OL)
        enh.append(1 if OPT/ENH > 1 else OPT/ENH)

    data = {
        'Online\nalgorithm': ol,
        'Adaptive\nenhanced\nonline': enh,
        'Two-phase\n(a=0.2m)': im,
        'Two-phase\n(a=0.5m)': im1,
        'Two-phase\n(a=0.8m)': im2,
    }
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'm={m}_SCPF={SCPF}_addPF={addPF}_enhanced.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
for i in range(3):
    Box_picmaker_enhanced(100, 1000)    

# (Swarm Plot) - Enhanced
def Swarm_picmaker_enhanced(line, time):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # New list for enhanced algorithm
    opt = []

    for i in range(time):
        # m= 57
        # m = random.randint(10, 20)
        m = random.randint(10, 1000)
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000
        TL, AL = TaskGenerator(m)
        
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
        IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
        IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
        
        opt.append(OPT)
        
        # Calculate ratios for comparison
        im.append(1 if OPT/IM > 1 else OPT/IM)
        im1.append(1 if OPT/IM1 > 1 else OPT/IM1)
        im2.append(1 if OPT/IM2 > 1 else OPT/IM2)
        ol.append(1 if OPT/OL > 1 else OPT/OL)
        enh.append(1 if OPT/ENH > 1 else OPT/ENH)

    data = {
        'Online\nalgorithm': ol,
        'Adaptive\nenhanced\nonline': enh,
        'Two-phase\n(a=0.2m)': im,
        'Two-phase\n(a=0.5m)': im1,
        'Two-phase\n(a=0.8m)': im2,
        
    }
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    sns.swarmplot(data=df, size=4)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('enhanced_comparison_swarm.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
Swarm_picmaker_enhanced(100, 100)