#!/bin/bash
# $1 is the number of workers
# $2 is the optimizer of model
# $3 is the targted_loss of model
# $4 is the tensorflow port
# $5 is the empoch
# $6 batch size
# ps.sh run in ssd42

get_ps_conf(){
    ps=""
    for(( i=42; i > 2; i-- ))
    do
	if [ $i -lt 10 ]
	then
            ps=$ps",ssd0"$i":2222"
	else
	    ps=$ps",ssd"$i":2222"
	fi
    done
        ps=${ps:1}
};

get_worker_conf(){
    worker=""
    for(( i=42; i > 7; i-- ))
    do
	port=2222
	for(( j=0; j<$1; j++ ))
	do
            worker=$worker",ssd"$i":"$port
	    port=`expr $port + 1`
	done
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
kill_cluster_pid.sh 3 15 $1

get_ps_conf $1 $5
echo $ps
get_worker_conf $1 $2 $5
echo $worker
for(( i=42; i>2; i-- ))
do
{
    index=`expr 15 - $i`
    if [ $i == 42 ]
    then
	python /root/code/disCNN_cifar/disCNN_cifar.py $ps $worker --job_name=ps --task_index=$i
    else
	ip=""
	if [ $i -lt 10 ]
	then
            $ip="ssd0"$i
	else
	    $ip="ssd"$i
	fi
	ssh $ip python /root/code/disCNN_cifar/disCNN_cifar.py $ps $worker --job_name=ps --task_index=$index
	
    fi
    
}&
done

sum_index=`expr $1 \* 13`
index=0
for(( i=0; i>$sum_index; i++ ))
do
{
    idx=expr $i % 13`
    idx2=expr 15 % $idx`
    ip=""
    if [ $idx2 -lt 10 ]
    then
        $ip="ssd0"$idx2
    else
	$ip="ssd"$idx2
    fi
    if [ $idx2 == 42 ]
    do
	if [ $index != 0 ]
	then
	    sleep 0.5
	fi
	python /root/code/disAlexNet/disAlexNet.py $ps $worker --job_name=worker --task_index=$i --targted_accuracy=$3 --empoch=$4 --optimizer=$2 --Batch_size=$5 >> /root/code/$index".temp"
    else
	if [ $index != 0 ]
	then
	    sleep 0.5
	fi
	ssh $ip python /root/code/disAlexNet/disAlexNet.py $ps $worker --job_name=worker --task_index=$i --targted_loss=$3 --empoch=$4 --optimizer=$2 --Batch_size=$5 >> /root/code/$index".temp"
    done
    echo "worker"$index" complated"
    echo "1">temp$index
}&
done

while true
do
    flag=0
    for(( i=0; i<$sum_index; i++ ))
    do
    {   
	tem=`cat temp$i`
	flag=`expr $tem + $flag`
    }
    done	
    if [ $flag == $sum_index ]
    then
    	kill_cluster_pid2.sh 16 42 $1
	break
    fi
done 
rm -f temp*
rm -f /root/code/*.temp
echo "work done"
