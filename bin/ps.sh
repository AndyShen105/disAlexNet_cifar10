#!/bin/bash
# $1 is the number of PSs
# $2 is the number of workers
# $3 is the optimizer of model
# $4 is the learning_rate
# $5 is port
# $6 is the model
# $7 is the n_intra
# $8 is the batch_size
# $9 n_partition
# #10 n_features

# ps.sh run in b1g72

get_ps_conf(){
    ps=""
    for(( i=72; i > 72-$1; i-- ))
    do
        ps="$ps,b1g$i:"$2
    done
        ps=${ps:1}
};

get_worker_conf(){
    worker=""
    for(( i=72-$1; i > 72-$1-$2; i-- ))
    do
        worker="$worker,b1g$i:"$3
    done
    worker=${worker:1}
};

for(( i=0; i<$2; i++ ))
do
{
    echo "0">temp$i
}
done

echo "release port occpied!"
kill_cluster_pid.sh 37 72 $5

get_ps_conf $1 $5
echo $ps
get_worker_conf $1 $2 $5
echo $worker

for(( i=72; i>72-$1-$2; i-- ))
do
{
    if [ $i == 72 ]
    then
        python /root/code/disCNN_cifar/disCNN_cifar10.py $ps $worker --job_name=ps --task_index=0
    else
	ip=""
        if [ $i -lt 10 ]
        then
            ip="b1g"$i
        n=`expr 72 - $1`
        if [ $i -gt $n ]
        then
            index=`expr 72 - $i`
            ssh $ip python /root/code/disCNN_cifar/disCNN_cifar10.py $ps $worker --job_name=ps --task_index=$index >> temp$index
        else
            index=`expr 72 - $1 - $i`
	    if [ $index != 0 ]
	    then
		sleep 0.5
	    fi
	    ssh $ip python /root/code/disCNN_cifar/disCNN_cifar10.py $ps $worker --job_name=worker --task_index=$index --targted_loss=0.05 --batch_size=$8 --learning_rate=$4 --optimizer=$3 --n_intra_threads=$7 --n_partitions=$9 >> /root/code/$index".temp"
            echo "worker"$index" complated"
	    echo "1">temp$index
	fi
    fi
}&
done

while true
do
    flag=0
    for(( i=0; i<$2; i++ ))
    do
    {   
	tem=`cat temp$i`
	flag=`expr $tem + $flag`
    }
    done	
    if [ $flag == $2 ]
    then
    	kill_cluster_pid.sh 37 72 $5
	break
    fi
done 
rm -f temp*
rm -f /root/code/*.temp
echo "work done"
