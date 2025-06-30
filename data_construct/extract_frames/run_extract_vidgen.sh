cd /mnt/myworkspace/xic_space/projects/UniReal.open/data_construct/extract_frames

/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 1 &
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 2 &  
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 3 &  
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 4 &  
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 5 &  
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 6 & 
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 7 &
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 8 &
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 9 &
/mnt/myworkspace/condaenvs/flux/bin/python extract_vidgen.py -N 10 -k 10 &
wait
