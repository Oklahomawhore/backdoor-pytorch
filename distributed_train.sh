#!/bin/bash
# Usage: ./distributed_train.sh id0 id1 ... [arguments for main.py]
# Example:
# ./distributed_train.sh 0 3 --config ./configs/cifar100.yaml
# will launch  torch distributed run of train.py on GPUs 0,3
args=("$@")
re='^[0-9]+$'
COUNT=0
TOTAL=$#
i=0
declare -a DEVICES
declare -a ARGS
PARSE_DEVICE=1

# Add device ids into DEVICES and
# arguments after that into ARGS
while [[ $i -le $TOTAL ]]; do
	if [[ $PARSE_DEVICE = "1" ]]; then
		echo "parsing_argument $i: ${args[i]}"
		if [[ ${args[i]} =~ [0-9] ]]; then
			((COUNT=COUNT+1)); 
			DEVICES+=("${args[$i]}");
		else 
			echo "${args[i]} is not number. pass"
			ARGS+=("${args[$i]}")
			PARSE_DEVICE=0;
		fi
	else
		echo "added argument $i: ${args[i]}"
		ARGS+=("${args[$i]}")
	
	fi
	((i++));
	shift 1
done


# Join list by delimitor
function join_by {
	local d=${1-} f=${2-}
	if shift 2; then
		printf %s "$f" "${@/#/$d}"
	fi
}

echo "parsed devices: ${DEVICES[@]} total count: $COUNT"
echo "parsed args: (${ARGS[@]})"

if [[ "${#COUNT[@]}" = "0" ]]; then
	python main.py "$@";
else	
	CUDA_VISIBLE_DEVICES=$(join_by , "${DEVICES[@]}") torchrun --nproc_per_node=$COUNT train.py "${ARGS[@]}"
fi

