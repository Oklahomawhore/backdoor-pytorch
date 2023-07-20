id=$1
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

while [ $free_mem -lt 24500 ]; do
	free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
	sleep 5
done

CUDA_VISIBLE_DEVICES=2,3,4  torchrun --nproc_per_node=3 --master_addr=127.0.0.1 --master_port=39503 train.py --config ./configs/tiny-imagenet_resnet.yaml
