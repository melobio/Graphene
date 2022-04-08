----------------------------------------------------------------------

* O V E R V I E W * 

----------------------------------------------------------------------

We provide two python programs together with the publication:

Uncovering Disease-Disease Relationships Through The Human Interactome

by Joerg Menche, Amitabh Sharma, Maksim Kitsak, Susan Dina Ghiassian,
   Marc Vidal, Joseph Loscalzo & Albert-Laszlo Barabasi

- localization.py: This program determines the network-based
  localization for a given set of nodes on a given network

- seperation.py: This program determines the network-based distance
  and separation for two given sets of nodes on given network 

- three additional files are provided:
  + interactome.tsv: Table with the human interactome
  + MD.txt: File containing genes associated with multiple sclerosis
  + PD.txt: File containing genes associated with peroxisomal
    disorders

----------------------------------------------------------------------

* I N S T A L L A T I O N * 

----------------------------------------------------------------------

* REQUIREMENTS * 

You will need a working Python installation. The code has been tested
and should run under versions 2.4-2.7

The code uses the following packages that are part of the Python
Standard Library and should already be available on your system:
- sys
- optparse 
- random   

The following external packages may need to be installed separately:
- networkx
- numpy  

* SETUP * 

Simply put all files in a directory. On unix based systems, you can
make it executable by typing the following line in your command
window:

chmod +x localization.py separation.py

Afterwards you should be able to execute the code by simply typing the
following line in your command window:

./localization.py 

or 

./separation.py 

otherwise the following should always work:

python localization.py
python separation.py

------------------------------------------------------------

* U S A G E * 

------------------------------------------------------------
 
The two programs require external files as input:

- A file containing the underlying network as a tab-separated table.
  If non is given, the provided interactome.tsv will be used.

- One (localization.py) or two (separation.py) files containing a list
  of genes. See MS.txt and PD.txt for examples.

For more details on required and optional parameters type:

./localization.py --usage
./localization.py --help

and

./separation.py --usage
./separation.py --help




