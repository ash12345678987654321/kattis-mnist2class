import os
import random
import numpy as np

training=[]
training_ans=[]
validation=[]
validation_ans=[]

weights=2*np.random.rand(30,51)-1


sign = np.vectorize(lambda t: 1 if t>0 else -1)

def test(epoch_num):
    #out=sign(weights)
    out=weights
    
    file=open(os.getcwd()+"\\outputs\\epoch"+str(epoch_num)+".txt","w")
    for i in range(30):
        for j in range(51):
            file.write(str(out[i, j])+" ")
        file.write("\n")

    file.close()

    correct=0

    for k in range(len(validation)):
        layer=sign(np.matmul(out,validation[k]))

        zeros=0
        ones=0

        for i in range(0,30):
            if (i<15):
                zeros+=layer[i]
            else:
                ones+=layer[i]

        if (zeros>=ones and validation_ans[k]==0):
            correct+=1
        elif (zeros<ones and validation_ans[k]==1):
            correct+=1

    
    print("epoch "+str(epoch_num)+": "+str(correct)+"/"+str(len(validation)))
    

def main():
    global training,training_ans,validation,validation_ans,weights
    
    file=open("train.txt","r")
    for i in file.read().split("\n"):
        i=list(map(int,i.split(" ")))
        training.append(i[0:51])
        training_ans.append(i[51])
    file.close()
    
    file=open("validation.txt","r")
    for i in file.read().split("\n"):
        i=list(map(int,i.split(" ")))
        validation.append(i[0:51])
        validation_ans.append(i[51])
    file.close()

    validation=np.array(validation)
    validation_ans=np.array(validation_ans)

    training=np.array(training)
    training_ans=np.array(training_ans)

    test(0)

    #for each output node the goal is to reach GOAL or -GOAL depending on answer

    BATCH_SIZE=10000 #make this divisible by 10000
    LEARNING_RATE=0.00000001
    GOAL=10
    for epoch_num in range(1,1000):
        for BATCH in range(len(training)//BATCH_SIZE):
            gradient=np.zeros(shape=(30,51))
            cost=np.zeros(shape=(30,1))
            for NUM in range(BATCH*BATCH_SIZE,(BATCH+1)*BATCH_SIZE):
                layer=np.matmul(weights,training[NUM])
                gradient[0]=training[NUM]

                for i in range(0,30):
                    if (i<15 and training_ans[NUM]==0) or (15<=i and training_ans[NUM]==1):
                        cost[i,0]=(10-layer[i])**2
                    else:
                        cost[i,0]=-(-10-layer[i])**2;

                gradient+=cost*training[NUM]
                
            weights+=gradient*LEARNING_RATE
            
        test(epoch_num)
    
        

if __name__=="__main__":
    main()
