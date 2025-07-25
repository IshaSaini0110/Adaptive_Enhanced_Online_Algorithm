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

class AdaptiveOnlineScheduler:
    def __init__(self, m, initial_theta=100, min_theta=50, max_theta=200):
        self.m = m  # Number of instances
        self.theta = initial_theta  # Dynamic threshold
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.job_history = []  # Track job completion times
        self.completed_jobs = []  # Track completed job actual times

    def update_theta(self, completed_jobs):
        """Adjust theta based on median job completion time."""
        if completed_jobs:
            self.completed_jobs.extend(completed_jobs)
            # Keep only recent history (last 20 completions)
            if len(self.completed_jobs) > 20:
                self.completed_jobs = self.completed_jobs[-20:]
            
            median_time = np.median(self.completed_jobs)
            self.theta = np.clip(median_time * 0.8, self.min_theta, self.max_theta)

    def calculate_priority(self, job):
        """Calculate priority (Delta) for a job."""
        t_lower, t_upper = job['t_lower'], job['t_upper']
        if t_upper - t_lower > 0:
            return (t_upper - self.theta) * (t_upper + t_lower) / (t_upper - t_lower)
        else:
            return (t_upper - self.theta) * (t_upper + t_lower) / 1

    def schedule(self, jobs):
        """Priority scheduling with dynamic theta."""
        # Update theta based on completed jobs
        completed_actual_times = [job['actual_time'] for job in jobs if job.get('completed', False)]
        self.update_theta(completed_actual_times)

        # Calculate priorities for all jobs
        for job in jobs:
            if not job.get('completed', False):
                job['Delta'] = self.calculate_priority(job)
            else:
                job['Delta'] = -float('inf')  # Completed jobs have lowest priority

        # Sort by Delta (descending) and allocate instances
        active_jobs = [job for job in jobs if not job.get('completed', False)]
        active_jobs.sort(key=lambda x: x['Delta'], reverse=True)

        return active_jobs


def adaptive_scheduler_online(TL, PF, addPF, SCPF, line):
    """
    Adaptive online algorithm using the AdaptiveOnlineScheduler class.
    
    Args:
        TL: Task list [(lower_bound, upper_bound, actual_time), ...]
        PF: Base parallelization factor
        addPF: Additional parallelization factor per extra machine
        SCPF: Scheduling coordination penalty factor
        line: Threshold line (used as initial theta)
    """
    
    # Initialize scheduler
    scheduler = AdaptiveOnlineScheduler(m=len(TL), initial_theta=line)
    
    # Convert task list to job format
    jobs = []
    for i, (lb, ub, actual) in enumerate(TL):
        jobs.append({
            'id': i,
            't_lower': lb,
            't_upper': ub,
            'actual_time': actual,
            'allocated_instances': 1,  # Start with 1 machine
            'completed': False,
            'remaining_time': actual,
            'Delta': 0
        })
    
    total_runtime = 0
    available_machines = len(TL) - len(jobs)  # Available machines for allocation
    
    # Simulation loop
    while any(not job['completed'] for job in jobs):
        # Update current remaining times and upper bounds
        for job in jobs:
            if not job['completed']:
                k = job['allocated_instances']
                parallel_factor = PF + addPF * (k - 1)
                # Update upper bound based on current allocation
                job['t_upper'] = job['remaining_time'] * parallel_factor / k + SCPF * job['actual_time']
        
        # Use scheduler to prioritize jobs
        active_jobs = scheduler.schedule(jobs)
        
        # Calculate effective remaining time for each active job
        effective_times = []
        for job in active_jobs:
            k = job['allocated_instances']
            parallel_factor = PF + addPF * (k - 1)
            effective_time = job['remaining_time'] * parallel_factor / k + SCPF * job['actual_time']
            effective_times.append(effective_time)
        
        if not effective_times:
            break
            
        min_effective_time = min(effective_times)
        
        # Progress all active jobs by minimum effective time
        newly_completed = []
        for job in active_jobs:
            k = job['allocated_instances']
            parallel_factor = PF + addPF * (k - 1)
            
            # Calculate how much actual work gets done
            work_done = min_effective_time * k / parallel_factor
            job['remaining_time'] -= work_done
            
            if job['remaining_time'] <= 0:
                job['completed'] = True
                job['remaining_time'] = 0
                newly_completed.append(job)
                # Free up machines
                available_machines += job['allocated_instances']
        
        total_runtime += min_effective_time
        
        # Allocate freed machines to high-priority jobs
        if available_machines > 0:
            # Get prioritized active jobs
            prioritized_jobs = scheduler.schedule(jobs)
            
            for job in prioritized_jobs:
                if available_machines <= 0:
                    break
                
                # Only allocate to jobs that are above threshold and not completed
                if (not job['completed'] and 
                    job['t_upper'] > scheduler.theta and 
                    job['allocated_instances'] < len(TL)):
                    
                    job['allocated_instances'] += 1
                    available_machines -= 1
    
    print(f"Adaptive Scheduler Online Algorithm Final Runtime: {total_runtime}")
    return total_runtime

# def adaptive_enhanced_online(TL, PF, addPF, SCPF, line):
#     # Represent each task with original time, remaining time, allocated cores, and overhead count.
#     jobs = []
#     for lb, ub, actual in TL:
#         jobs.append({'original': actual, 'remaining': actual, 'allocated': 1, 'overhead': 0})
#     # Repeatedly assign extra cores to the job with highest positive benefit.
#     while True:
#         best_job = None
#         best_benefit = 0.0
#         best_predicted = 0.0
#         for job in jobs:
#             # Skip finished jobs or those not eligible (below threshold).
#             if job['remaining'] <= 0 or job['original'] <= line:
#                 continue
#             k = job['allocated']
#             predicted = job['remaining'] * (k/(k+1)) * (1 + addPF)
#             benefit = (job['remaining'] - predicted) - SCPF * job['original']
#             if benefit > best_benefit:
#                 best_benefit = benefit
#                 best_job = job
#                 best_predicted = predicted
#         if best_job is None or best_benefit <= 0:
#             break
#         # Allocate one more core to the best job.
#         best_job['allocated'] += 1
#         best_job['remaining'] = best_predicted
#         best_job['overhead'] += 1
#     # After allocation, compute each job's final completion time including overhead.
#     final_times = []
#     for job in jobs:
#         final_times.append(job['remaining'] + job['overhead'] * SCPF * job['original'])
#     total_runtime = max(final_times) if final_times else 0.0
#     print(f"Adaptive Enhanced Online Algorithm Final Runtime: {total_runtime}")
#     return total_runtime


# def Line_picmaker_enhanced(line, length):
#     im = []
#     im1 = []
#     im2 = []
#     ol = []
#     enh = []  # Existing enhanced algorithm
#     adp = []  # New adaptive scheduler algorithm
#     opt = []

#     for i in range(length):
#         m = random.randint(10, 1000)
#         PF = 1
#         addPF = (random.randint(1, 500)) / 1000
#         SCPF = (random.randint(1, 500)) / 1000

#         TL, AL = TaskGenerator(m)
       
#         OPT = optimal(AL, PF, addPF, SCPF)
#         OL = online(TL, PF, addPF, SCPF, line)
#         ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
#         ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
#         IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
#         opt.append(OPT)
       
#         # Calculate ratios for comparison
#         im.append(1 if OPT/IM > 1 else OPT/IM)
#         im1.append(1 if OPT/IM1 > 1 else OPT/IM1)
#         im2.append(1 if OPT/IM2 > 1 else OPT/IM2)
#         ol.append(1 if OPT/OL > 1 else OPT/OL)
#         enh.append(1 if OPT/ENH > 1 else OPT/ENH)
#         adp.append(1 if OPT/ADP > 1 else OPT/ADP)  # New algorithm ratio

#     x = range(length)
#     plt.figure(figsize=(10, 3))
#     plt.plot(x, im, color='#FF8C00', label='Two-phase algorithm (a=0.2m)', marker='^', markersize=3)
#     plt.plot(x, im1, color='#00BFFF', label='Two-phase algorithm (a=0.5m)', marker='v', markersize=3)
#     plt.plot(x, im2, color='#808080', label='Two-phase algorithm (a=0.8m)', marker='s', markersize=3)
#     plt.plot(x, ol, color='r', label='Online algorithm', marker='.', markersize=3)
#     plt.plot(x, enh, color='#32CD32', label='Adaptive Enhanced Online', marker='o', markersize=3)
#     plt.plot(x, adp, color='#9370DB', label='Adaptive Scheduler Online', marker='d', markersize=3)  # New line
   
#     plt.yticks(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.35),  # Adjusted to accommodate new legend
#         ncol=3,
#         columnspacing=1.2,
#         handletextpad=0.5,
#         borderaxespad=0,
#         fontsize=12,
#         frameon=False)
    
#     plt.savefig('enhanced_with_scheduler.pdf', dpi=600, format='pdf', bbox_inches='tight')
#     plt.show()
#     plt.clf()


# def Bar_picmaker_enhanced(line, time):
#     im = []
#     im1 = []
#     im2 = []
#     ol = []
#     enh = []  # Existing enhanced algorithm
#     adp = []  # New adaptive scheduler algorithm
#     opt = []

#     m = random.randint(10, 1000)
#     TL, AL = TaskGenerator(m)
#     bar = []
   
#     for i in range(time):
#         PF = 1
#         addPF = (random.randint(1, 500)) / 1000
#         SCPF = (random.randint(1, 500)) / 1000
       
#         OPT = optimal(AL, PF, addPF, SCPF)
#         OL = online(TL, PF, addPF, SCPF, line)
#         ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
#         ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
#         IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
#         opt.append(OPT)
#         ol.append(OL)
#         enh.append(ENH)
#         adp.append(ADP)  # New algorithm results
#         im.append(IM)
#         im1.append(IM1)
#         im2.append(IM2)
#         bar.append(i)
   
#     bar_width = 0.10  # Adjusted width for 7 bars
#     index_opt = np.arange(len(bar))
#     index_OL = index_opt + bar_width
#     index_enh = index_opt + bar_width*2
#     index_adp = index_opt + bar_width*3  # New algorithm position
#     index_IM = index_opt + bar_width*4
#     index_IM1 = index_opt + bar_width*5
#     index_IM2 = index_opt + bar_width*6
   
#     plt.figure(figsize=(14, 6))  # Increased figure width
#     plt.bar(index_opt, height=opt, width=bar_width, color='k', label='Offline optimal algorithm')
#     plt.bar(index_OL, height=ol, width=bar_width, color='r', label='Online algorithm')
#     plt.bar(index_enh, height=enh, width=bar_width, color='#32CD32', label='Adaptive enhanced online')
#     plt.bar(index_adp, height=adp, width=bar_width, color='#9370DB', label='Adaptive scheduler online')  # New bar
#     plt.bar(index_IM, height=im, width=bar_width, color='#FF8C00', label='Two-phase algorithm (a=0.2m)')
#     plt.bar(index_IM1, height=im1, width=bar_width, color='#00BFFF', label='Two-phase algorithm (a=0.5m)')
#     plt.bar(index_IM2, height=im2, width=bar_width, color='#808080', label='Two-phase algorithm (a=0.8m)')
   
#     plt.xticks(index_opt + bar_width*6 / 2, bar)  # Adjusted center position
#     plt.yticks(fontsize=14)
#     plt.ylabel('Time (second)', fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.xlabel('Groups', fontsize=14)
#     plt.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.25),  # Adjusted for more legend items
#         ncol=3,  # Keep 3 columns but will have more rows
#         borderaxespad=0,
#         columnspacing=1.2,
#         handletextpad=0.5,
#         fontsize=11,  # Slightly smaller font
#         frameon=False)
#     ax = plt.gca()
#     ax.ticklabel_format(style='plain', axis='y')
#     plt.tight_layout()
#     plt.savefig(f'm={m}_enhanced_with_scheduler.pdf', dpi=600, format='pdf', bbox_inches='tight')
#     plt.show()
#     plt.clf()

# for i in range(3):
#     Bar_picmaker_enhanced(100, 15)


# def Box_picmaker_enhanced(line, time):
#     im = []
#     im1 = []
#     im2 = []
#     ol = []
#     enh = []  # Existing enhanced algorithm
#     adp = []  # New adaptive scheduler algorithm
#     opt = []

#     m = random.randint(10, 1000)
#     PF = 1
#     addPF = (random.randint(1, 500)) / 1000
#     SCPF = (random.randint(1, 500)) / 1000

#     for i in range(time):
#         TL, AL = TaskGenerator(m)
       
#         OPT = optimal(AL, PF, addPF, SCPF)
#         OL = online(TL, PF, addPF, SCPF, line)
#         ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
#         ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
#         IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
#         opt.append(OPT)
       
#         # Calculate ratios for comparison
#         im.append(1 if OPT/IM > 1 else OPT/IM)
#         im1.append(1 if OPT/IM1 > 1 else OPT/IM1)
#         im2.append(1 if OPT/IM2 > 1 else OPT/IM2)
#         ol.append(1 if OPT/OL > 1 else OPT/OL)
#         enh.append(1 if OPT/ENH > 1 else OPT/ENH)
#         adp.append(1 if OPT/ADP > 1 else OPT/ADP)  # New algorithm ratio

#     data = {
#         'Online\nalgorithm': ol,
#         'Adaptive\nenhanced\nonline': enh,
#         'Adaptive\nscheduler\nonline': adp,  # New algorithm data
#         'Two-phase\n(a=0.2m)': im,
#         'Two-phase\n(a=0.5m)': im1,
#         'Two-phase\n(a=0.8m)': im2,
#     }
   
#     df = pd.DataFrame(data)
#     plt.figure(figsize=(14, 6))  # Increased figure width
    
#     # Create box plot with custom colors
#     box_plot = df.plot.box(
#         color=dict(boxes='black', whiskers='black', medians='red', caps='black'),
#         patch_artist=True,
#         ax=plt.gca()
#     )
    
#     # Set custom colors for each box
#     colors = ['#FF0000', '#32CD32', '#9370DB', '#FF8C00', '#00BFFF', '#808080']
#     for patch, color in zip(box_plot.artists, colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.7)
    
#     plt.grid(linestyle="--", alpha=0.3)
#     plt.xticks(fontsize=12)  # Adjusted font size
#     plt.yticks(fontsize=14)
#     plt.ylabel('Performance Ratio', fontsize=14)
#     plt.tight_layout()
#     plt.savefig(f'm={m}_SCPF={SCPF}_addPF={addPF}_enhanced_with_scheduler.pdf', 
#                 dpi=600, format='pdf', bbox_inches='tight')
#     plt.show()
#     plt.clf()

# for i in range(3):
#     Box_picmaker_enhanced(100, 1000)


# def Swarm_picmaker_enhanced(line, time):
#     im = []
#     im1 = []
#     im2 = []
#     ol = []
#     enh = []  # Existing enhanced algorithm
#     adp = []  # New adaptive scheduler algorithm
#     opt = []

#     for i in range(time):
#         m = random.randint(10, 1000)
#         PF = 1
#         addPF = (random.randint(1, 500)) / 1000
#         SCPF = (random.randint(1, 500)) / 1000
#         TL, AL = TaskGenerator(m)
       
#         OPT = optimal(AL, PF, addPF, SCPF)
#         OL = online(TL, PF, addPF, SCPF, line)
#         ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
#         ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
#         IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
#         IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
#         opt.append(OPT)
       
#         # Calculate ratios for comparison
#         im.append(1 if OPT/IM > 1 else OPT/IM)
#         im1.append(1 if OPT/IM1 > 1 else OPT/IM1)
#         im2.append(1 if OPT/IM2 > 1 else OPT/IM2)
#         ol.append(1 if OPT/OL > 1 else OPT/OL)
#         enh.append(1 if OPT/ENH > 1 else OPT/ENH)
#         adp.append(1 if OPT/ADP > 1 else OPT/ADP)  # New algorithm ratio

#     data = {
#         'Online\nalgorithm': ol,
#         'Adaptive\nenhanced\nonline': enh,
#         'Adaptive\nscheduler\nonline': adp,  # New algorithm data
#         'Two-phase\n(a=0.2m)': im,
#         'Two-phase\n(a=0.5m)': im1,
#         'Two-phase\n(a=0.8m)': im2,
#     }
   
#     df = pd.DataFrame(data)
#     plt.figure(figsize=(14, 6))  # Increased figure width
    
#     # Create swarm plot with custom colors
#     colors = ['#FF0000', '#32CD32', '#9370DB', '#FF8C00', '#00BFFF', '#808080']
#     sns.swarmplot(data=df, size=4, palette=colors)
    
#     plt.tick_params(labelsize=12)  # Adjusted font size
#     plt.ylabel('Performance Ratio', fontsize=14)
#     plt.tight_layout()
#     plt.savefig('enhanced_comparison_swarm_with_scheduler.pdf', 
#                 dpi=600, format='pdf', bbox_inches='tight')
#     plt.show()
#     plt.clf()

# Swarm_picmaker_enhanced(100, 100)

def Line_picmaker_enhanced(line, length):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # Existing enhanced algorithm
    adp = []  # New adaptive scheduler algorithm
    opt = []

    for i in range(length):
        # m = 20
        # m = random.randint(10, 300)
        m = random.randint(10, 1000)
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000

        TL, AL = TaskGenerator(m)
       
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
        IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
        IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
        IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
        opt.append(OPT)
       
        # Calculate ratios for comparison
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

        if (OPT / ADP > 1):
            adp.append(1)
        else:
            adp.append(OPT / ADP)

    x = range(length)
    plt.figure(figsize=(10, 3))
    plt.plot(x, im, color='#FF8C00', label='Two-phase algorithm (a=0.2m)', marker='^', markersize=3)
    plt.plot(x, im1, color='#00BFFF', label='Two-phase algorithm (a=0.5m)', marker='v', markersize=3)
    plt.plot(x, im2, color='#808080', label='Two-phase algorithm (a=0.8m)', marker='s', markersize=3)
    plt.plot(x, ol, color='r', label='Online algorithm', marker='.', markersize=3)
    plt.plot(x, enh, color="#32CD32", label='Adaptive Enhanced Online', marker='o', markersize=3)
    plt.plot(x, adp, color="#A132CD", label='Adaptive Online algorithm', marker='d', markersize=3)
   
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

def Bar_picmaker_enhanced(line, time):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # Existing enhanced algorithm
    adp = []  # New adaptive scheduler algorithm
    opt = []
    
    # m = 20
    # m = random.randint(10, 300)
    m = random.randint(10, 1000)
    TL, AL = TaskGenerator(m)
    bar = []
   
    for i in range(time):
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000
       
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
        IM = intermediate(AL, m, int(0.2*m), PF+SCPF*(m-1), addPF, SCPF)
        IM1 = intermediate(AL, m, int(0.5*m), PF+SCPF*(m-1), addPF, SCPF)
        IM2 = intermediate(AL, m, int(0.8*m), PF+SCPF*(m-1), addPF, SCPF)
       
        opt.append(OPT)
        ol.append(OL)
        enh.append(ENH)
        adp.append(ADP)
        im.append(IM)
        im1.append(IM1)
        im2.append(IM2)
        bar.append(i)
   
    bar_width = 0.10
    index_opt = np.arange(len(bar))
    index_OL = index_opt + bar_width
    index_adp = index_opt + bar_width*2
    index_enh = index_opt + bar_width*3
    index_IM = index_opt + bar_width*4
    index_IM1 = index_opt + bar_width*5
    index_IM2 = index_opt + bar_width*6
   
    plt.figure(figsize=(12, 5))
    plt.bar(index_opt, height=opt, width=bar_width, color='k', label='Offline optimal algorithm')
    plt.bar(index_OL, height=ol, width=bar_width, color='r', label='Online algorithm')
    plt.bar(index_enh, height=enh, width=bar_width, color='#32CD32', label='Adaptive enhanced online')
    plt.bar(index_adp, height=adp, width=bar_width, color='#A132CD', label='Adaptive online algorithm')
    plt.bar(index_IM, height=im, width=bar_width, color='#FF8C00', label='Two-phase algorithm (a=0.2m)')
    plt.bar(index_IM1, height=im1, width=bar_width, color='#00BFFF', label='Two-phase algorithm (a=0.5m)')
    plt.bar(index_IM2, height=im2, width=bar_width, color='#808080', label='Two-phase algorithm (a=0.8m)')
   
    plt.xticks(index_opt + bar_width*6 / 2, bar)
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


def Box_picmaker_enhanced(line, time):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # Existing enhanced algorithm
    adp = []  # New adaptive scheduler algorithm
    opt = []

    # m = 20
    # m = random.randint(10, 300)
    m = random.randint(10, 1000)
    PF = 1
    addPF = (random.randint(1, 500)) / 1000
    SCPF = (random.randint(1, 500)) / 1000

    for i in range(time):
        TL, AL = TaskGenerator(m)
       
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
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
        adp.append(1 if OPT/ADP > 1 else OPT/ADP)

    data = {
        'Online\nalgorithm': ol,
        'Adaptive\nonline': adp,
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

def Swarm_picmaker_enhanced(line, time):
    im = []
    im1 = []
    im2 = []
    ol = []
    enh = []  # Existing enhanced algorithm
    adp = []  # New adaptive scheduler algorithm
    opt = []

    for i in range(time):
        # m = 20
        # m = random.randint(10, 300)
        m = random.randint(10, 1000)
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000
        TL, AL = TaskGenerator(m)
       
        OPT = optimal(AL, PF, addPF, SCPF)
        OL = online(TL, PF, addPF, SCPF, line)
        ENH = adaptive_enhanced_online(TL, PF, addPF, SCPF, line)
        ADP = adaptive_scheduler_online(TL, PF, addPF, SCPF, line)  # New algorithm
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
        adp.append(1 if OPT/ADP > 1 else OPT/ADP)

    data = {
        'Online\nalgorithm': ol,
        'Adaptive\nonline': adp,
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
