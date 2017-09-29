for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.05 2222 20000 100
    i=`expr 1 + $i`
done

for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.05 2222 20000 300
    i=`expr 1 + $i`
done

for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.05 2222 20000 500
    i=`expr 1 + $i`
done

for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.05 2222 20000 700
    i=`expr 1 + $i`
done

for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.05 2222 20000 900
    i=`expr 1 + $i`
done

for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.95 2222 20 0.0001 11
    i=`expr 1 + $i`
done

for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.95 2222 20 0.0001 13
    i=`expr 1 + $i`
done

for(( i=1; i < 27; i++ ))
do
    j=`expr 27 - $i`
    ./ps.sh $i $j Adam 0.95 2222 20 0.0001 15
    i=`expr 1 + $i`
done
