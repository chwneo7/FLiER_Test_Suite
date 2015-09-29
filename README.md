# FLiER_Test_Suite

This repository contains code to solve power flow equations, as well as to run
the FLiER algorithms discussed in our paper.

Some example python calls, to be run in the /Code directory, are the following:


python FLiER_Test.py -network ../Networks/ieee57cdf.txt -pmus 3,12,33 -use_filter True -noise 0.0 -write_file output.txt

python FLiER_Substation_Test.py -network ../Networks/ieee57cdf.txt -pmus 3,12,33 -use_filter True -noise 0.0 -test_type Full -write_file output.txt

python FLiER_Substation_Test.py -network ../Networks/ieee57cdf.txt -pmus 3,12,33 -use_filter True -noise 0.0 -test_type Single_Lines -write_file output.txt


Note that our code 0-indexes network buses, while e.g. MATLAB 1-indexes buses. So, for example, placing PMUs on 0-indexed buses 3, 12, and 33 means placing PMUs on 1-indexed buses 4, 13, and 34.