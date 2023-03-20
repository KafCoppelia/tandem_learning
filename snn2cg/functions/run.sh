# connectionRate1=(0.0 0.05 0.1 0.15 0.2)
# rateName1=(0 5 10 15 20)

connectionRate1=(0.85 0.95)
rateName1=(85 95)

# connectionRate2=(0.25 0.3 0.4 0.5 0.6)
# rateName2=(25 30 40 50 60)

# connectionRate3=(0.7 0.75 0.8 0.9 1.002)
# rateName3=(70 75 80 90 100)

inputData=255
config=ASIC_LongRun

for ((i=0;i<${#connectionRate1[@]};++i))
do
    printf "[%d] inputWidth %d, rate %f\n" $i $inputData ${connectionRate1[$i]}
    {
        python functions/buildCores.py --input $inputData --rate ${connectionRate1[$i]} --config $config  >output/log$i.txt 2>&1 & 
    }
done
wait

# for ((i=0;i<${#connectionRate2[@]};++i))
# do
#     printf "[%d] inputWidth %d, rate %f\n" $i $inputData ${connectionRate2[$i]}
#     {
#         python functions/buildCores.py --input $inputData --rate ${connectionRate2[$i]} --config $config  >output/log$i.txt 2>&1 & 
#     }
# done
# wait


# for ((i=0;i<${#connectionRate3[@]};++i))
# do
#     printf "[%d] inputWidth %d, rate %f\n" $i $inputData ${connectionRate3[$i]}
#     {
#         python functions/buildCores.py --input $inputData --rate ${connectionRate3[$i]} --config $config  >output/log$i.txt 2>&1 & 
#     }
# done
# wait

for ((i=0;i<${#connectionRate1[@]};++i))
do
    rate=${rateName1[$i]}
    d=${config}_${inputData}_${rate}
    echo $d
    cd output 
    rm "$d/test3.txt" 
    tar -zcvf long_${rate}.tar.gz $d/ &
    cd -
done
wait

# for ((i=0;i<${#connectionRate2[@]};++i))
# do
#     rate=${rateName2[$i]}
#     d=${config}_${inputData}_${rate}
#     echo $d
#     cd output 
#     rm "$d/test3.txt" 
#     tar -zcvf long_${rate}.tar.gz $d/ &
#     cd -
# done
# wait

# for ((i=0;i<${#connectionRate3[@]};++i))
# do
#     rate=${rateName3[$i]}
#     d=${config}_${inputData}_${rate}
#     echo $d
#     cd output 
#     rm "$d/test3.txt" 
#     tar -zcvf long_${rate}.tar.gz $d/ &
#     cd -
# done
# wait

