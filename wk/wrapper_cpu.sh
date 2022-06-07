#!/bin/bash

ID=${OMPI_COMM_WORLD_LOCAL_RANK}
if [ ${ID} -eq 0 ]; then
    mnode=0
else
    mnode=2
fi

numactl -m ${mnode} $@
