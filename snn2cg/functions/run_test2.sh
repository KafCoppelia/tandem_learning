
# configName1=("ASIC_All_LCN1" "ASIC_All_LCN2" "ASIC_All_LCN4" "ASIC_All_LCN8")

# configName2=("ASIC_None_LCN1" "ASIC_None_LCN2" "ASIC_None_LCN4" "ASIC_None_LCN8")

# configName1=("ASIC_All_LCN16" "ASIC_All_LCN32" "ASIC_All_LCN64")

configName2=("ASIC_None_LCN16" "ASIC_None_LCN32" "ASIC_None_LCN64")
inputData=255
rate=1.005

# for ((i=0;i<${#configName1[@]};++i))
# do
#     printf "[%d] inputWidth %d, config %s\n" $i $inputData ${configName1[$i]}
#     {
#         python functions/buildCores.py --input $inputData --rate ${rate} --config ${configName1[$i]}  >output/log$i.txt 2>&1 & 
#     }
# done
# wait

for ((i=0;i<${#configName2[@]};++i))
do
    printf "[%d] inputWidth %d, config %s\n" $i $inputData ${configName2[$i]}
    {
        python functions/buildCores.py --input $inputData --rate ${rate} --config ${configName2[$i]}  >output/log${i}_.txt 2>&1 & 
    }
done
wait


# for ((i=0;i<${#connectionRate3[@]};++i))
# do
#     printf "[%d] inputWidth %d, rate %f\n" $i $inputData ${connectionRate3[$i]}
#     {
#         python functions/buildCores.py --input $inputData --rate ${connectionRate3[$i]} --config $config  >output/log$i.txt 2>&1 & 
#     }
# done
# wait

# for ((i=0;i<${#configName1[@]};++i))
# do
#     d=${configName1[$i]}_${inputData}_100
#     echo $d
#     cd output 
#     # rm "$d/test3.txt" 
#     tar -zcvf all_${i}.tar.gz $d/ &
#     cd -
# done
# wait

for ((i=0;i<${#configName2[@]};++i))
do
    d=${configName2[$i]}_${inputData}_100
    echo $d
    cd output 
    # rm "$d/test3.txt" 
    tar -zcvf none_${i}.tar.gz $d/ &
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

