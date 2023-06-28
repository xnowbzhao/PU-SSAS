ROOT=..


BUILD_PATH=$ROOT/data/ShapeNet.build
OUTPUT_PATH=$ROOT/data/ShapeNet

NPROC=8
TIMEOUT=180
N_VAL=100
N_TEST=100
N_AUG=50

declare -a CLASSES=(
02691156  
02828884  
02933112  
03001627  
03211117  
03636649  
03691459   
04256520
04379243
04401088
04530566
)

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}

lsfilter2() {
 folder=$1
 ext=$2

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $folder/$filename$ext ] && [ ! -d $folder/$filename$ext ]; then
    echo $filename
   fi
 done
}
