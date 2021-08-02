TOPLEVEL := $(or $(TOPLEVEL),..)
ABS_TOPLEVEL=$(abspath $(TOPLEVEL))

cellicium/build:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose build cellicium

cellicium/up:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose up cellicium

cellicium/prebuilt:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose -f docker-compose.yaml -f docker-compose.prebuilt.yaml up --build cellicium

cellicium/service:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose up -d cellicium

cellicium/bash:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose exec cellicium /bin/bash

cellicium/push:
	echo 'docker image ls docker_cellicium'
	echo 'docker tag CONTAINERID naskomit/cellicium:1.x'
	echo 'docker push naskomit/cellicium:1.x'
