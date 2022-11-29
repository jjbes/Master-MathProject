Math Project
========

Author/Contact: Harold Mouchère (harold.mouchere@univ-nantes.fr)
Edited by : Julien Aubert (julien.aubert---beduchaud@etu.univ-nantes.fr)

This is a Master Project to apply machine (deep) learning tool on a challenging computer vision problem: recognition of handwritten Math Expression.

The subject is available in the directory "subject" and the provided code in "code". Some toy examples are available in the "data" directory but the complete dataset is available in the TC11 repository.


## Usage of some provided tools

### the docker machine with CROHME tools + pytorch 

The provided docker file ``` Dockerfile ``` is built with command 

``` docker build -t mathprj . ```

It is staked with the machine provided here :  https://gitlab.univ-nantes.fr/mouchere-h/DockerMachines
 built with the command line "docker build -t crohme ."
 
Then start the machine by sharing the current working directory : 

``` docker run -rm -v `pwd`:/home/work -it mathprj ```

### convertInkmlToImg.py
	Usage: convertInkmlToImg.py  (path to file or folder) dim padding
		+ (file|folder)        - required str
		+ dim                  - optional int (def = 28)
		+ padding              - optional int (def =  0)

### processAll.sh
Apply all recognition step to all inkml files and produce the LG files

Usage: processAll <input_inkml_dir> <output_lg_dir>

###  python3 segmenter.py 
Generate from an inkml file hypotheses of symbol in a LG file

usage: python3 segmenter.py [-i fname][-o fname][-s N]
     -i fname / --inkml fname  : input file name (inkml file)
     -o fname / --output fname : output file name (LG file)
     -s N / --str N            : if no inkmlfile is selected, run with N strokes
	 
### python3 segmentSelect.py
Keep or not each segment hypotheses and generate a new LG file 

usage: python3 [-o fname] [-s] segmentSelect.py inkmlfile lgFile
     inkmlfile  : input inkml file name
     lgFile     : input LG file name
     -o fname / --output fname : output file name (LG file)
     -s         : save hyp images
	 
### python3 symbolReco.py

Recognize each hypothesis and save all acceptable recognition in a LG file

usage: python3 symbolReco.py [-s] [-o fname][-w weigthFile] inkmlfile lgFile
     inkmlfile  : input inkml file name
     lgFile     : input LG file name
     -o fname / --output fname : output file name (LG file)
     -w fname / --weight fname : weight file name (nn pytorch file)
     -s         : save hyp images
	 
###  python3 selectBestSeg.py

From an LG file with several hypotheses, keep only one coherent global solution (greedy sub-optimal)

usage: python3 selectBestSeg.py  [-o fname] lgFile
     lgFile     : input LG file name
     -o fname / --output fname : output file name (LG file)
	 
### ./listExistingLG.sh
Preparation of the partial evaluation of recognition. Not needed if all expressions are recognized.
Generate the list of existing LG files with associated LG ground-thruth.

Usage: processAll <ground-truthdir> <output_lg_dir>

### evaluate (from LgEval lib)

Compare output LG files with ground-truth LG file and generate a detailed summary of metrics.

2 Usages: global evaluation or partial evaluation from a list of couple
       evaluate outputDir groundTruthDir 
       evaluate fileList 