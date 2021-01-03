all: unRle gpuCompressor

unRle:
	$(MAKE) -C extractors
	mv extractors/unRle .

gpuCompressor:
	$(MAKE) -C compressor
	mv compressor/gpuCompressor .

clean:
	$(MAKE) -C compressor clean
	$(MAKE) -C extractors clean
	-rm gpuCompressor unRle
