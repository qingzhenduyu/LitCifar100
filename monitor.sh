while true
do
  # shellcheck disable=SC1060
  # stat1-stat8 对应显卡数量
  # $09表示gpustat查询到的显卡占用显存
  # '2p-9p' 表示显卡号 '2p' 对应第一张显卡，依此类推
  stat3=$(gpustat | awk '{print $12}' | sed -n '2p')
  stat4=$(gpustat | awk '{print $12}' | sed -n '3p')
  stat5=$(gpustat | awk '{print $12}' | sed -n '4p')
  stat6=$(gpustat | awk '{print $12}' | sed -n '6p')
  stat_arr=($stat3 $stat4 $stat5 $stat6)
  gpu_available=0
  gpu_available_index_arr=()
  # 得到空闲GPU的数量和对应的序号
  for i in ${!stat_arr[@]}
  do
     # 如果显存占用小于阈值(单位M)，继续
    if [ "${stat_arr[$i]}" -lt 300 ]
    then
      gpu_available=$[gpu_available+1]
      gpu_available_index_arr[${#gpu_available_index_arr[@]}]=$i
    else
      echo "${stat_arr[$i]}"
    fi
  done
  echo '-可用GPU数:'$gpu_available', 第'${gpu_available_index_arr[@]}'块GPU可用'
  # 如果GPU数大于指定数量，取指定数量GPU开始训练
  if [ $gpu_available -ge 4 ]
  then
    echo 'start running the code...'
    # 传值操作,即需要运行的代码脚本
    python -u train.py --config configs/vit.yaml --model.pretrained False --trainer.max_epochs 200 --model.learning_rate 1e-3 
    break # 防止下一次循环又重复运行上一行命令
  fi
  sleep 180 # 单位秒
done
