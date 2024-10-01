TIMESTAMP := $(shell date +%m-%d-%H-%M-%S)

clean-logs:
	mkdir -p archive
	-tar -czvf "archive/join-logs-$(TIMESTAMP).tar.gz" join*
	-tar -czvf "archive/rocksdb_join-logs-$(TIMESTAMP).tar.gz" join*
	-tar -czvf "archive/merged-logs-$(TIMESTAMP).tar.gz" merged*
	-tar -czvf "archive/rocksdb_merged-logs-$(TIMESTAMP).tar.gz" merged*
	-tar -czvf "archive/base-logs-$(TIMESTAMP).tar.gz" base*
	-tar -czvf "archive/rocksdb_base-logs-$(TIMESTAMP).tar.gz" base*
	rm -rf join*
	rm -rf merged*
	rm -rf base*
	rm -rf rocksdb_join*
	rm -rf rocksdb_merged*
	rm -rf rocksdb_base*

clean-plots:
	mkdir -p archive
	tar -czvf "archive/plots-$(TIMESTAMP).tar.gz" plots*
	rm -rf plots*

clean-all: clean-logs clean-plots

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

in-paper:
	python3 main.py
	python3 main.py --suffix=-sel19
	python3 main.py --dram_gib=16 --in_memory=1
	python3 main.py --rocksdb=1
	python3 main.py --type=selectivity
	python3 main.py --rocksdb=1 --type=selectivity
	python3 AggPlotter.py --stats=update_manual.csv

copy-plots:
	mkdir -p in-paper
	cp plots/all-tx/TXs-s.png in-paper/all-tx.png
	cp plots/all-tx/all-tx_size.png in-paper/all-tx-size.png
	cp plots/all-tx-sel19/TXs-s.png in-paper/all-tx-sel19.png
	cp plots/all-tx-rocksdb/TXs-s.png in-paper/all-tx-rocksdb.png
	cp plots/in-memory-all-tx/TXs-s.png in-paper/all-tx-in-memory.png
	cp plots/selectivity/TXs-s-read-locality-size.png in-paper/selectivity-read-locality-size.png
	cp plots/selectivity/TXs-s-scan-size.png in-paper/selectivity-scan-size.png
	cp plots/selectivity/selectivity_size.png in-paper/selectivity-size.png
	cp plots/selectivity-rocksdb/TXs-s-read-locality-size.png in-paper/selectivity-rocksdb-read-locality-size.png
	cp plots/selectivity-rocksdb/selectivity_size.png in-paper/selectivity-rocksdb-size.png
	cp plots/selectivity-rocksdb/selectivity_time.png in-paper/selectivity-rocksdb-time.png
	cp plots/update_manual/TXs-s-manual.png in-paper/update-manual.png

.PHONY: in-paper move-plots