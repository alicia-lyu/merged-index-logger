TIMESTAMP := $(shell date +%m-%d-%H-%M-%S)

tar-all:
	tar -czvf "all-plots-$(TIMESTAMP).tar.gz" plots*

clean-logs:
	mkdir -p archive
	tar -czvf "archive/join-logs-$(TIMESTAMP).tar.gz" join*
	tar -czvf "archive/merged-logs-$(TIMESTAMP).tar.gz" merged*
	rm -rf join*
	rm -rf merged*

clean-plots:
	mkdir -p archive
	tar -czvf "archive/plots-$(TIMESTAMP).tar.gz" plots*
	rm -rf plots*

clean-all: clean-logs clean-plots

in-memory:
	python3 main.py --dram_gib=16 --type=all-tx --suffix=-sel19 --in_memory=True

selectivity:
	python3 main.py --dram_gib=0.3 --type=selectivity

same-size:
	python3 main.py --dram_gib=0.3 --type=all-tx --suffix=-sel19

# Directory pattern that you want to check
DIRS := $(wildcard */)

# Rule to check and clean directories
clean_unfinished:
	@for dir in $(DIRS); do \
		if [ -f $$dir/*sum.csv ]; then \
			row_count=$$(awk 'END {print NR}' $$dir/*sum.csv); \
			if [ $$row_count -lt 100 ]; then \
				echo "Removing $$dir (rows: $$row_count)"; \
				rm -rf $$dir; \
			fi \
		fi \
	done

default:
	python3 main.py --type=all-tx --dram=0.3

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