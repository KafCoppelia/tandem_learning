data_sets=(1 4 16 32 64 128 256 512 768 1024)

inputData=255
config=ASIC_LongRun

for ((i=0;i<${#data_sets[@]};++i))
do
    printf "[%d] inputWidth %d, core num %d\n" $i $inputData ${data_sets[$i]}
    {
        python functions/buildCores.py --input $inputData --rate 1.0 --connection_rate 1.0 --config $config --num ${data_sets[$i]} >output/log$i.txt 2>&1 & 
    }
done
wait

for ((i=0;i<${#data_sets[@]};++i))
do
    num=${data_sets[$i]}
    d=${config}_${inputData}_100_connect_100_${num}
    echo $d
    cd output 
    tar -zcvf long_core_${num}.tar.gz $d/ &
    cd -
done
wait