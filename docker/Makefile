TOPLEVEL := $(or $(TOPLEVEL),..)
ABS_TOPLEVEL=$(abspath $(TOPLEVEL))

cellicium/build:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose build cellicium

cellicium/up:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose up cellicium

cellicium/prebuilt:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose -f docker-compose.yaml -f docker-compose.prebuilt.yaml up --build cellicium

cellicium/bash:
	export ABS_TOPLEVEL=$(ABS_TOPLEVEL) && docker-compose exec cellicium /bin/bash
