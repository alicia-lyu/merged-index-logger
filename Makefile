TIMESTAMP := $(shell date +%m-%d-%H-%M-%S)

tar-all:
	tar -czvf "all-plots-$(TIMESTAMP).tar.gz" plots*

in-memory:
	python3 main.py --dram_gib=2 --target_gib=1 --type=all-tx --suffix=-sel20 --in_memory=True

plot-all:
	python3 main.py --type=read
	python3 main.py --type=scan
	python3 main.py --type=write
	python3 main.py --type=update-size
	python3 main.py --type=selectivity
	python3 main.py --type=included-columns
	python3 plot_size.py

all: plot-all tar-all

.PHONY: tar-all, plot-all