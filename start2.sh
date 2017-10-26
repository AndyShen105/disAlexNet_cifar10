for(( i=2; i < 16; i++ ))
do  
    n_inter_op=`expr 16 - $i`
    ./ps.sh 2 25 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 4 23 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 6 21 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 8 19 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 10 17 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 12 15 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 14 13 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 16 11 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 18 9 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 20 7 Adam 0.5 2222 200 $i $n_inter_op
done

for(( i=2; i < 16; i++ ))
do
    n_inter_op=`expr 16 - $i`
    ./ps.sh 22 5 Adam 0.5 2222 200 $i $n_inter_op
done
