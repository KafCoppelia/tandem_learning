connectionRate1=(0.0 0.1 0.2 0.3)
rateName1=(0 10 20 30)

connectionRate2=(0.4 0.5 0.6 0.7)
rateName2=(40 50 60 70)

connectionRate3=(0.8 0.9 1.0)
rateName3=(80 90 100)

inputData=255
config=ASIC_All_connect

for ((i=0;i<${#connectionRate1[@]};++i))
do
    printf "[%d] inputWidth %d, rate %f\n" $i $inputData ${connectionRate1[$i]}
    {
        python functions/buildCores.py --input $inputData --connection_rate ${connectionRate1[$i]} --config $config  >output/log$i.txt 2>&1 & 
    }
done
wait

for ((i=0;i<${#connectionRate2[@]};++i))
do
    printf "[%d] inputWidth %d, rate %f\n" $i $inputData ${connectionRate2[$i]}
    {
        python functions/buildCores.py --input $inputData --connection_rate ${connectionRate2[$i]} --config $config  >output/log$i.txt 2>&1 & 
    }
done
wait


for ((i=0;i<${#connectionRate3[@]};++i))
do
    printf "[%d] inputWidth %d, rate %f\n" $i $inputData ${connectionRate3[$i]}
    {
        python functions/buildCores.py --input $inputData --connection_rate ${connectionRate3[$i]} --config $config  >output/log$i.txt 2>&1 & 
    }
done
wait

for ((i=0;i<${#connectionRate1[@]};++i))
do
    rate=${rateName1[$i]}
    d=${config}_${inputData}_100_connect_${rate}
    echo $d
    cd output 
    tar -zcvf long_${rate}.tar.gz $d/ &
    cd -
done
wait

for ((i=0;i<${#connectionRate2[@]};++i))
do
    rate=${rateName2[$i]}
    d=${config}_${inputData}_100_connect_${rate}
    echo $d
    cd output  
    tar -zcvf long_${rate}.tar.gz $d/ &
    cd -
done
wait

for ((i=0;i<${#connectionRate3[@]};++i))
do
    rate=${rateName3[$i]}
    d=${config}_${inputData}_100_connect_${rate}
    echo $d
    cd output 
    tar -zcvf long_${rate}.tar.gz $d/ &
    cd -
done
wait

