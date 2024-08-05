TIMESTAMP := $(shell date +%m-%d-%H-%M-%S)

tar-all:
	tar -czvf "all-plots-$(TIMESTAMP).tar.gz" plots*

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